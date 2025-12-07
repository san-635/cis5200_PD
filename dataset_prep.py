""" Prepares the PPMI_datscan_mri dataset, converts DICOMs to JPEGs, and splits into train and test sets. """

import os
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom
from PIL import Image

from config import config

def select_subject_image_ids(orig_ppmi_metadata):
    """ Selects subjects based on the following criteria:
        - Select one (Modality='SPECT', Type='Pre-processed') and one (Modality='MRI', Type='Original') per subject.
        - Ignore subjects that do not have both modalities.
        - For each subject that has more than one such pair, choose the one with the closest acquisition dates. """
    orig_ppmi_metadata_df = pd.read_csv(orig_ppmi_metadata)
    ppmi_datscan_mri = []

    # select appr. subjects and their image IDs
    print("Selecting appropriate subjects and their image IDs")
    grouped = orig_ppmi_metadata_df.groupby('Subject')
    for subject_id, group in grouped:
        spect_entries = group[(group['Modality'] == 'SPECT') & (group['Type'] == 'Pre-processed')]
        mri_entries = group[(group['Modality'] == 'MRI') & (group['Type'] == 'Original')]

        if not spect_entries.empty and not mri_entries.empty:
            min_date_diff = pd.Timedelta.max
            earliest_spect = None
            earliest_mri = None

            for _, spect_row in spect_entries.iterrows():
                spect_date = pd.to_datetime(spect_row['Acq Date'])
                for _, mri_row in mri_entries.iterrows():
                    mri_date = pd.to_datetime(mri_row['Acq Date'])
                    date_diff = abs(spect_date - mri_date)
                    if date_diff < min_date_diff:
                            min_date_diff = date_diff
                            earliest_spect = spect_row
                            earliest_mri = mri_row

            ppmi_datscan_mri.append({
                'subject_ID': subject_id,
                'spect_image_ID': earliest_spect['Image Data ID'],
                'mri_image_ID': earliest_mri['Image Data ID']
            })

    return ppmi_datscan_mri

def exclude_3d_mri(orig_ppmi_dir, ppmi_datscan_mri, orig_ppmi_metadata):
    """ Exclude subjects whose MRI images are 3D volumes (i.e., have more than one DICOM slice). """
    excl_mri_images = []
    for subject in ppmi_datscan_mri:
        mri_dicom_files = glob.glob(os.path.join(orig_ppmi_dir, str(subject['subject_ID']), '**', f"{subject['mri_image_ID']}", '*.dcm'), recursive=True)
        sample_mri_dicom = pydicom.dcmread(mri_dicom_files[0])
        if len(sample_mri_dicom.pixel_array.shape) != 2:
            excl_mri_images.append(subject['mri_image_ID'])

    ppmi_datscan_mri = [subject for subject in ppmi_datscan_mri if subject['mri_image_ID'] not in excl_mri_images]

    # build final metadata for PPMI_datscan_mri with only the selected rows
    orig_ppmi_metadata_df = pd.read_csv(orig_ppmi_metadata)
    ppmi_datscan_mri_metadata = orig_ppmi_metadata_df[(orig_ppmi_metadata_df['Subject'].isin([s['subject_ID'] for s in ppmi_datscan_mri])) & 
                                                      ((orig_ppmi_metadata_df['Image Data ID'].isin([s['spect_image_ID'] for s in ppmi_datscan_mri])) |
                                                       (orig_ppmi_metadata_df['Image Data ID'].isin([s['mri_image_ID'] for s in ppmi_datscan_mri])))]

    print(f"Extra note: {len(excl_mri_images)} subjects with 3D MRI images are excluded")
    return ppmi_datscan_mri, ppmi_datscan_mri_metadata

def save_ppmi_datscan_mri(ppmi_datscan_mri, orig_ppmi_dir, final_ppmi_dir):
    """ Copies the selected image DICOM files into 'PPMI_datscan_mri' directory, whose expected structure is:
        PPMI_datscan_mri/
            <subject_ID_1>/
                SPECT_<spect_image_ID>/
                    <DICOM file>
                MRI_<mri_image_ID>/
                    <DICOM file>
            <subject_ID_2>/
                ...
            metadata.csv
    """
    for subject in tqdm(ppmi_datscan_mri):
        subject_id = str(subject['subject_ID'])
        spect_id = str(subject['spect_image_ID'])
        mri_id = str(subject['mri_image_ID'])

        subject_dir = os.path.join(orig_ppmi_dir, subject_id)
        spect_src = glob.glob(os.path.join(subject_dir, '**', f'*{spect_id}*'), recursive=True)[0]
        mri_src = glob.glob(os.path.join(subject_dir, '**', f'*{mri_id}*'), recursive=True)[0]

        spect_src_file = os.path.join(spect_src, os.listdir(spect_src)[0])
        mri_src_file = os.path.join(mri_src, os.listdir(mri_src)[0])

        spect_dst = os.path.join(final_ppmi_dir, f"{subject_id}", f"SPECT_{spect_id}")
        mri_dst = os.path.join(final_ppmi_dir, f"{subject_id}", f"MRI_{mri_id}")
        os.makedirs(spect_dst, exist_ok=True)
        os.makedirs(mri_dst, exist_ok=True)

        shutil.copy2(spect_src_file, spect_dst)
        shutil.copy2(mri_src_file, mri_dst)
    
    print("Saved PPMI_datscan_mri dataset to", final_ppmi_dir)

def print_save_stats(ppmi_datscan_mri, ppmi_datscan_mri_metadata, stats_path, print_stats=True):
    """ Saves (and optionally prints) some statistics about the PPMI_datscan_mri dataset. """
    lines = []

    def log(s=""):
        if print_stats:
            print(s)
        lines.append(str(s))

    log(f"Total subjects: {len(ppmi_datscan_mri)}")
    log(f"Total subjects (M): {ppmi_datscan_mri_metadata[ppmi_datscan_mri_metadata['Sex'] == 'M']['Subject'].nunique()}")
    log(f"Total subjects (F): {ppmi_datscan_mri_metadata[ppmi_datscan_mri_metadata['Sex'] == 'F']['Subject'].nunique()}")
    total_age_groups = ppmi_datscan_mri_metadata.groupby(pd.cut(ppmi_datscan_mri_metadata['Age'], range(30, 91, 10)), observed=True)['Subject'].nunique()
    log("Total subjects by age:")
    for interval, count in total_age_groups.items():
        log(f"  {str(interval):<8} -> {count}")

    log(f"Total PD subjects: {ppmi_datscan_mri_metadata[ppmi_datscan_mri_metadata['Group'] == 'PD']['Subject'].nunique()}")
    log(f"Total PD subjects (M): {ppmi_datscan_mri_metadata[(ppmi_datscan_mri_metadata['Group'] == 'PD') & (ppmi_datscan_mri_metadata['Sex'] == 'M')]['Subject'].nunique()}")
    log(f"Total PD subjects (F): {ppmi_datscan_mri_metadata[(ppmi_datscan_mri_metadata['Group'] == 'PD') & (ppmi_datscan_mri_metadata['Sex'] == 'F')]['Subject'].nunique()}")
    pd_age_groups = ppmi_datscan_mri_metadata[ppmi_datscan_mri_metadata['Group'] == 'PD'].groupby(pd.cut(ppmi_datscan_mri_metadata['Age'], range(30, 91, 10)), observed=True)['Subject'].nunique()
    log("Total PD subjects by age:")
    for interval, count in pd_age_groups.items():
        log(f"  {str(interval):<8} -> {count}")

    log(f"Total Control subjects: {ppmi_datscan_mri_metadata[ppmi_datscan_mri_metadata['Group'] == 'Control']['Subject'].nunique()}")
    log(f"Total Control subjects (M): {ppmi_datscan_mri_metadata[(ppmi_datscan_mri_metadata['Group'] == 'Control') & (ppmi_datscan_mri_metadata['Sex'] == 'M')]['Subject'].nunique()}")
    log(f"Total Control subjects (F): {ppmi_datscan_mri_metadata[(ppmi_datscan_mri_metadata['Group'] == 'Control') & (ppmi_datscan_mri_metadata['Sex'] == 'F')]['Subject'].nunique()}")
    control_age_groups = ppmi_datscan_mri_metadata[ppmi_datscan_mri_metadata['Group'] == 'Control'].groupby(pd.cut(ppmi_datscan_mri_metadata['Age'], range(30, 91, 10)), observed=True)['Subject'].nunique()
    log("Total Control subjects by age:")
    for interval, count in control_age_groups.items():
        log(f"  {str(interval):<8} -> {count}")

    with open(stats_path, "w") as f:
        f.write("\n".join(lines) + "\n")

def train_test_split(ppmi_datscan_mri, ppmi_datscan_mri_metadata, train_ratio=0.8, random_seed=42):
    """ Splits the dataset's metadata into train (80%) and test (20%) datasets. """
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(ppmi_datscan_mri))
    train_size = int(len(ppmi_datscan_mri) * train_ratio)

    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]

    train_set = [ppmi_datscan_mri[i] for i in train_indices]
    test_set = [ppmi_datscan_mri[i] for i in test_indices]

    train_metadata = ppmi_datscan_mri_metadata[(ppmi_datscan_mri_metadata['Subject'].isin([s['subject_ID'] for s in train_set])) & 
                                                      ((ppmi_datscan_mri_metadata['Image Data ID'].isin([s['spect_image_ID'] for s in train_set])) |
                                                       (ppmi_datscan_mri_metadata['Image Data ID'].isin([s['mri_image_ID'] for s in train_set])))]
    test_metadata = ppmi_datscan_mri_metadata[(ppmi_datscan_mri_metadata['Subject'].isin([s['subject_ID'] for s in test_set])) & 
                                                      ((ppmi_datscan_mri_metadata['Image Data ID'].isin([s['spect_image_ID'] for s in test_set])) |
                                                       (ppmi_datscan_mri_metadata['Image Data ID'].isin([s['mri_image_ID'] for s in test_set])))]
    return train_metadata, test_metadata

def mri_check(dataset_folder):
    """ Checks if the MRI DICOM images are 2D or 3D. """
    dicom_files = glob.glob(os.path.join(dataset_folder, '*.dcm'))
    slices, file_names = [], []
    for file in tqdm(dicom_files, desc="Reading DICOM files"):
        slices.append(pydicom.dcmread(file))
        file_names.append(os.path.basename(file))

    print(f"Number of MRI DICOM files found: {len(slices)}")
    print(f"Attributes in a DICOM file: {slices[0].dir()}")

    # check if all slices have the same dimensions
    slice_shapes = [s.pixel_array.shape for s in slices]
    if len(set(slice_shapes)) != 1:
        print("Error: Not all DICOM slices have the same dimensions.")
        print(f"Slice shapes found: {set(slice_shapes)}")
        print(f"Number of slices with each shape: { {shape: slice_shapes.count(shape) for shape in set(slice_shapes)} }")

    # check if all slices have 'ImagePositionPatient' attribute
    if not all(hasattr(s, 'ImagePositionPatient') for s in slices):
        print("Error: Some DICOM slices are missing the 'ImagePositionPatient' attribute.")
        print(f"Number of slices with 'ImagePositionPatient': {sum(hasattr(s, 'ImagePositionPatient') for s in slices)} out of {len(slices)}")

def mri_dicom_to_jpg(dicom_folder):
    """ Converts a 2D DICOM slice into a single 2D JPEG image. """
    dicom_path = glob.glob(os.path.join(dicom_folder, '*.dcm'))[0]
    slice = pydicom.dcmread(dicom_path).pixel_array
    # no sorting needed for single series

    if slice.max() == 0:
        jpg_array = np.zeros_like(slice, dtype=np.uint8)
        # print(f"Warning: DICOM slice for {dicom_path.split(os.path.sep)[-3]} has all zero pixel values.")
    else:
        # normalize and rescale the image to 8-bit (0-255)
        norm_slice = (np.maximum(slice, 0) / slice.max()) * 255.0
        jpg_array = np.uint8(norm_slice)
        
    # save as PIL image and convert to numpy array
    pil_image = Image.fromarray(jpg_array)
    output_path = os.path.join(dicom_folder, (os.path.basename(dicom_path).replace('.dcm', '.jpg')))
    pil_image.save(output_path)

def datscan_dicom_to_jpg(dicom_folder):
    """ Converts a 3D DICOM series into a stack of 2D JPEG images. """
    dicom_path = glob.glob(os.path.join(dicom_folder, '*.dcm'))[0]
    slices = pydicom.dcmread(dicom_path).pixel_array
    # no sorting needed for single series

    # slices 40-42 are to be saved as a single 3D JPEG image
    slices_3d = slices[40:43]
    jpg_3d_array = np.zeros_like(slices_3d, dtype=np.uint8)
    for i, slice_2d in enumerate(slices_3d):
        if slice_2d.max() == 0:
            jpg_3d_array[i] = np.zeros_like(slice_2d, dtype=np.uint8)
            # print(f"Warning: DICOM slice {i} for {dicom_path.split(os.path.sep)[-3]} has all zero pixel values.")
        else:
            # normalize and rescale the image to 8-bit (0-255)
            norm_slice = (np.maximum(slice_2d, 0) / slice_2d.max()) * 255.0
            jpg_3d_array[i] = np.uint8(norm_slice)

    # save as PIL image and convert to numpy array
    pil_image = Image.fromarray(np.transpose(jpg_3d_array, (1, 2, 0)))
    output_path = os.path.join(dicom_folder, (os.path.basename(dicom_path).replace('.dcm', '.jpg')))
    pil_image.save(output_path)

def dicom_to_jpg(final_ppmi_dir):
    """ Converts and saves all DICOM files in PPMI_datscan_mri as JPEGs. """
    subject_dirs = [d for d in glob.glob(os.path.join(final_ppmi_dir, '*')) if os.path.isdir(d)]

    for subject_dir in tqdm(subject_dirs, desc="Converting DICOMs to JPEGs"):
        datscan_dir = glob.glob(os.path.join(subject_dir, 'SPECT_*'))[0]
        mri_dir = glob.glob(os.path.join(subject_dir, 'MRI_*'))[0]

        datscan_dicom_to_jpg(datscan_dir)
        mri_dicom_to_jpg(mri_dir)

def main():
    orig_ppmi_dir = os.path.join(config.PPMI_PARENT_PATH, 'PPMI')
    final_ppmi_dir = os.path.join(config.PPMI_PARENT_PATH, 'PPMI_datscan_mri')
    os.makedirs(final_ppmi_dir, exist_ok=True)

    # select appr. subjects and their image IDs
    orig_ppmi_metadata = os.path.join(orig_ppmi_dir, 'metadata.csv')
    ppmi_datscan_mri = select_subject_image_ids(orig_ppmi_metadata)
    ppmi_datscan_mri, ppmi_datscan_mri_metadata = exclude_3d_mri(orig_ppmi_dir, ppmi_datscan_mri, orig_ppmi_metadata)

    # form PPMI_datscan_mri dataset
    ppmi_datscan_mri_metadata.to_csv(os.path.join(final_ppmi_dir, 'metadata.csv'), index=False)
    save_ppmi_datscan_mri(ppmi_datscan_mri, orig_ppmi_dir, final_ppmi_dir)

    # print some stats
    stats_path = os.path.join(final_ppmi_dir, 'stats.txt')
    print_save_stats(ppmi_datscan_mri, ppmi_datscan_mri_metadata, stats_path, config.PRINT_STATS)

    # train-test split
    train_metadata, test_metadata = train_test_split(ppmi_datscan_mri, ppmi_datscan_mri_metadata, train_ratio=0.8, random_seed=config.SEED)
    train_metadata.to_csv(os.path.join(final_ppmi_dir, 'train_metadata.csv'), index=False)
    test_metadata.to_csv(os.path.join(final_ppmi_dir, 'test_metadata.csv'), index=False)

    # convert DICOM to JPEG
    dicom_to_jpg(final_ppmi_dir)

if __name__ == "__main__":
    main()