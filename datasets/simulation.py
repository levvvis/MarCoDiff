import os
import numpy as np
import tigre
from tigre.utilities import CTnoise
import tigre.algorithms as algs

def process_and_save(ct_image, geo, angles, output_path, niter=40):
    # Add noise to simulate low-dose CT projections
    noise_projections = CTnoise.add(ct_image, Poisson=1e5, Gaussian=np.array([0, 10], dtype=np.float32))
    print(f"Noise Projections shape: {noise_projections.shape}")

    # Save the noisy projections as input file
    # np.save(output_path, noise_projections)

    # Reconstruct using OS-SART algorithm to check if projections are valid
    imgOSSART = algs.ossart(noise_projections, geo, angles, niter)

    # Verify the shape of the reconstructed image
    imgOSSART = np.squeeze(imgOSSART)
    print(f"Reconstructed image shape2: {imgOSSART.shape}")
    np.save(output_path, imgOSSART)
    # Check min and max values of the reconstructed images
    print(f"OS-SART min-max: {imgOSSART.min()} - {imgOSSART.max()}")

def process_directory(directory):
    # Define TIGRE geometry for parallel beam
    geo = tigre.geometry()
    geo.DSD = 1536  # Distance Source Detector (mm)
    geo.DSO = 1000  # Distance Source Origin (mm)
    geo.nVoxel = np.array([1, 512, 512])  # number of voxels across 3 dimensions (vx)
    geo.sVoxel = np.array([1, 512, 512])  # total size of the image (mm)
    geo.dVoxel = geo.sVoxel / geo.nVoxel  # size of each voxel (mm)
    geo.nDetector = np.array([1, 1024])  # number of detector pixels (px)
    geo.dDetector = np.array([geo.dVoxel[0], 0.8])  # size of each pixel on the detector (mm)
    geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector (mm)
    geo.offOrigin = np.array([0, 0, 0])  # Offset of image from the origin (mm)
    geo.offDetector = np.array([0, 0])  # Offset of the Detector (mm)
    geo.mode = "parallel"  # Using parallel beam geometry

    # Define angles of projection
    angles = np.linspace(0, 2 * np.pi, 360)  # Increase the number of angles for higher resolution

    for filename in os.listdir(directory):
        if "target" in filename:
            # Load the target file
            filepath = os.path.join(directory, filename)
            target_data = np.load(filepath).astype(np.float32)  # Convert to float32
            print(f"Loaded {filename} with shape: {target_data.shape}")

            # Ensure the image format matches geo.nVoxel
            target_data = target_data[np.newaxis, :, :]  # Add a new axis to match [1, H, W]

            # Generate sinogram (projections) for parallel beam
            projections = tigre.Ax(target_data, geo, angles)

            # Create the new filename for the input file
            input_filename = filename.replace('target', 'input')
            input_filepath = os.path.join(directory, input_filename)

            # Process and save the noisy projections as input file
            process_and_save(projections, geo, angles, input_filepath, niter=100)

            print(f"Saved noisy projections to {input_filepath}")


directory = r''

process_directory(directory)

print("All files have been processed and saved.")