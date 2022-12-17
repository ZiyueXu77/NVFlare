from typing import Optional, Dict

from nvflare.apis.dxo import DataKind, from_shareable, DXO
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.app_constant import AppConstants


class SKLearnCollector(Aggregator):
    def __init__(self, aggregator_id: str):
        super().__init__()
        self.accumulator: Optional[Dict] = None
        self.aggregator_id = aggregator_id
        self.aggregator = None

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        if not self.aggregator:
            self.aggregator = fl_ctx.get_engine().get_component(self.aggregator_id)
        if not self.accumulator:
            self.accumulator = self.aggregator.get_accumulator()

        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        dxo = self._get_contribution(shareable, fl_ctx)
        if dxo is None or dxo.data is None:
            self.log_error(fl_ctx, "no data to aggregate")
            return False

        data = dxo.data
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        return self._accept_contribution(contributor_name, current_round, data, fl_ctx)

    def _client_in_accumulator(self, client_name):
        return client_name in self.accumulator

    def _accept_contribution(self,
                             contributor: str,
                             current_round: int,
                             data: dict,
                             fl_ctx: FLContext) -> bool:

        if not self._client_in_accumulator(contributor):
            self.accumulator[contributor] = self.aggregator.get_model_params(data)
            accepted = True
        else:
            self.log_info(fl_ctx,
                          f"Discarded: Current round: {current_round} " +
                          f"contributions already include client: {contributor}",
                          )
            accepted = False
        return accepted

    def _get_contribution(self, shareable: Shareable, fl_ctx: FLContext) -> Optional[DXO]:
        contributor_name = shareable.get_peer_prop(key=ReservedKey.IDENTITY_NAME, default="?")
        try:
            dxo = from_shareable(shareable)
        except BaseException:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return None

        rc = shareable.get_return_code()
        if rc and rc != ReturnCode.OK:
            self.log_warning(fl_ctx, f"Contributor {contributor_name} returned rc: {rc}. Disregarding contribution.")
            return None
        expected_data_kind = self.aggregator.get_expected_data_kind()
        if dxo.data_kind != expected_data_kind:
            self.log_error(fl_ctx, "expected {} but got {}".format(expected_data_kind, dxo.data_kind))
            return None

        contribution_round = shareable.get_header(AppConstants.CONTRIBUTION_ROUND)
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        if contribution_round != current_round:
            self.log_warning(fl_ctx, f"discarding DXO from {contributor_name} at round: "
                                     f"{contribution_round}. Current round is: {current_round}")
            return None

        return dxo

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        self.log_debug(fl_ctx, "Start aggregation")
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        site_num = len(self.accumulator)
        self.log_info(fl_ctx, f"aggregating {site_num} update(s) at round {current_round}")

        model = self.aggregator.aggregate(current_round, self.accumulator)
        # Reset accumulator for next round,
        self.aggregator.reset()
        self.log_debug(fl_ctx, "End aggregation")

        dxo = DXO(data_kind=self.aggregator.get_expected_data_kind(), data=model)
        return dxo.to_shareable()
