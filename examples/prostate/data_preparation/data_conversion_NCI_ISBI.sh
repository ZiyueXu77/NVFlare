for subset in 3T Dx; do
  target_folder=./dataset/NCI_ISBI_${subset}
  mkdir ${target_folder}
  mkdir ${target_folder}/Image
  mkdir ${target_folder}/Mask
  for split in Training; do
    source_img_folder=./Raw/NCI_ISBI/Image/${split}
    source_msk_folder=./Raw/NCI_ISBI/Mask/${split}
    find $source_img_folder -mindepth 1 -maxdepth 1 -type d | while read case; do
      case=$(basename ${case})
      echo ${case}
      if [[ "${case}" == *"${subset}"* ]]; then
        img_path=${source_img_folder}/${case}/*/*
        python3 utils/dicom_to_nifti.py --dicom_folder ${img_path} --nifti_path ${target_folder}/Image/${case}.nii.gz
        msk_path=${source_msk_folder}/${case}.nrrd
        python3 utils/nrrd_to_nifti.py --input_path ${msk_path} --reference_path ${target_folder}/Image/${case}.nii.gz --output_path ${target_folder}/Mask/${case}.nii.gz
        python3 utils/label_threshold.py --input_path ${target_folder}/Mask/${case}.nii.gz --output_path ${target_folder}/Mask/${case}.nii.gz
      fi
    done
  done
  for split in Test Leaderboard; do
    source_img_folder=./Raw/NCI_ISBI/Image/${split}
    source_msk_folder=./Raw/NCI_ISBI/Mask/${split}
    find $source_img_folder -mindepth 1 -maxdepth 1 -type d | while read case; do
      case=$(basename ${case})
      echo ${case}
      if [[ "${case}" == *"${subset}"* ]]; then
        img_path=${source_img_folder}/${case}/*/*
        python3 utils/dicom_to_nifti.py --dicom_folder ${img_path} --nifti_path ${target_folder}/Image/${case}.nii.gz
        msk_path=${source_msk_folder}/${case}_truth.nrrd
        python3 utils/nrrd_to_nifti.py --input_path ${msk_path} --reference_path ${target_folder}/Image/${case}.nii.gz --output_path ${target_folder}/Mask/${case}.nii.gz
        python3 utils/label_threshold.py --input_path ${target_folder}/Mask/${case}.nii.gz --output_path ${target_folder}/Mask/${case}.nii.gz
      fi
    done
  done
done

rm ${target_folder}/Image/ProstateDx-01-0055.nii.gz
rm ${target_folder}/Mask/ProstateDx-01-0055.nii.gz
