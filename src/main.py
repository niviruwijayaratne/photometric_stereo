import numpy as np
from pathlib import Path
import imageio
from utils import lRGB2XYZ
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D
from cp_hw5 import integrate_poisson, integrate_frankot, load_sources
from scipy.ndimage import gaussian_filter, uniform_filter
from tqdm import tqdm
import subprocess
import argparse as ap


def preprocess(image):
    # image = image[600:-750, 1930:-2100]  # mcdonalds2
    # image = image[975:-275, 2500:-2600]  # lintroller
    # image = uniform_filter(image, size=8)
    # image = image[850:-1600, 2100:-2100]  # lintroller2
    # image = image[850:-1400, 2100:-2100]  # contactbox
    # image = image[1025:-1575, 2100:-2100]  # contactbox2
    # image = image[900:-1485, 2100:-2100]  # cologne_cap
    # image = image[1150:-1575, 2100:-2100]  # decoration
    # image = image[400:-1575, 2100:-2300]  # kitchen stuff
    H, W, C = image.shape
    return image


def load_images(image_dir: Path, mask_path: Path, mydata=True) -> np.ndarray:
    """Loads input images, converts to XYZ color space, and stacks luminance channel into len(image_dir) x # of pixels matrix.

  Args:
    image_dir: Path to directory containing input images.
    mask_path: Path to mask if available
    mydata: Flag indicating whether using own data (implies a certain file structure)

  Returns:
    len(image_dir) x # of pixels per image matrix
   """
    if mydata:
        if mask_path is not None:
            mask = imageio.imread(mask_path)[..., -1]
            mask = np.expand_dims((mask - mask.min()) /
                                  (mask.max() - mask.min()), -1)
            mask = np.pad(mask, ((16, 0), (16, 0), (0, 0)))
            mask[np.where(mask)] = 1
            mask = mask.astype(bool)
        else:
            mask = 1

        im_paths = sorted(list(image_dir.rglob("*.tiff")))
        im_dirs = sorted(list(image_dir.glob("*")))
        im0 = preprocess(imageio.imread(im_paths[0]))
        H, W, C = im0.shape
        luminance_matrix = np.zeros(
            (len([x for x in im_dirs if x.is_dir()]), H * W))
        i = 0
        for im_dir in tqdm(im_dirs):
            if not im_dir.is_dir() or "uncalibrated_stereo" in str(im_dir.stem):
                continue
            im = np.zeros_like(im0)
            images = list(im_dir.glob("*.tiff"))
            num_images = len(images)
            for im_path in images:
                im = im + preprocess(imageio.imread(im_path)*mask)

            xyz = lRGB2XYZ(im/(num_images))
            luminance_matrix[i] = xyz[..., 1].reshape(1, -1)
            i += 1
    else:
        im_paths = sorted(list(image_dir.rglob("*.tif")))
        im0 = preprocess(imageio.imread(im_paths[0]))
        H, W, C = im0.shape
        luminance_matrix = np.zeros(
            (len(im_paths), H * W))
        for i, im_path in enumerate(im_paths):
            im = preprocess(imageio.imread(im_path))
            xyz = lRGB2XYZ(im)
            luminance_matrix[i] = xyz[..., 1].reshape(1, -1)

    return luminance_matrix


def visualize(normals=None, albedo=None, depth=None, sigma=3, GBR=[0, 0, 0], out_dir=None, plot=True):
    if out_dir is not None:
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=False)
    mu, nu, lam = GBR
    if plot:
        if normals is not None:
            normals = (normals + 1) / 2
            plt.imshow(normals)
            plt.show()
        if albedo is not None:
            plt.imshow(albedo, cmap='gray')
            plt.show()
        if depth is not None:
            plt.imshow(depth, cmap='gray')
            plt.show()

    if out_dir is not None:
        if not plot:
            if normals is not None:
                normals = (normals + 1) / 2
        albedo = (albedo - albedo.min())/(albedo.max() - albedo.min())
        if sigma is not None:
            if normals is not None:
                imageio.imwrite(out_dir / f"normals_{sigma}_{mu}_{nu}_{lam}.png",
                                (normals*255).astype(np.uint8))
            if albedo is not None:
                imageio.imwrite(
                    out_dir / f"albedo_{sigma}_{mu}_{nu}_{lam}.png", (np.squeeze(albedo)*255).astype(np.uint8))
            if depth is not None:
                imageio.imwrite(
                    out_dir / f"depth_{sigma}_{mu}_{nu}_{lam}.png", (depth*255).astype(np.uint8))
        else:
            if normals is not None:
                imageio.imwrite(out_dir / f"normals.png",
                                (normals*255).astype(np.uint8))
            if albedo is not None:
                imageio.imwrite(
                    out_dir / f"albedo.png", (np.squeeze(albedo)*255).astype(np.uint8))
            if depth is not None:
                imageio.imwrite(
                    out_dir / f"depth.png", (depth*255).astype(np.uint8))


def plot_surface(Z):
    # Z is an HxW array of surface depths
    H, W = Z.shape
    x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
    # set 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # add a light and shade to the axis for visual effect
    # (use the ‘-’ sign since our Z-axis points down)
    ls = LightSource()
    color_shade = ls.shade(-Z, plt.cm.gray)
    # display a surface
    # (control surface resolution using rstride and cstride)
    surf = ax.plot_surface(
        x, y, -Z, facecolors=color_shade, rstride=4, cstride=4)
    # turn off axis
    plt.axis('off')
    plt.show()


def uncalibrated_photometric_stereo(
        luminance_matrix: np.ndarray,
        enforce_integrability: bool = False, dims=(4016, 6016, 3), sigma=3, GBR=[0, 0, 1], Q=None) -> (np.ndarray, np.ndarray):

    mu, nu, lam = GBR
    GBR = np.array([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    U, S, Vt = np.linalg.svd(luminance_matrix, full_matrices=False)
    Le = U[:, :3] @ np.diag(np.sqrt(S[:3]))
    Be = np.diag(np.sqrt(S[:3])) @ Vt[:3]
    if not enforce_integrability and Q is not None:
        Le = (Q @ Le.T).T
        Be = np.linalg.inv(Q.T) @ Be
    H, W, C = dims
    Be_reshaped = Be.reshape((C, H, W)).transpose(1, 2, 0)
    if enforce_integrability:
        be1, be2, be3 = Be_reshaped[...,
                                    0], Be_reshaped[..., 1], Be_reshaped[..., 2]
        be1, be2, be3 = be1.reshape(-1, 1), be2.reshape(-1,
                                                        1), be3.reshape(-1, 1)
        Be_gauss = np.dstack(
            [gaussian_filter(Be_reshaped[..., i], sigma=sigma) for i in range(3)])

        dbe_dx = np.gradient(Be_gauss, axis=1)
        dbe_dx1, dbe_dx2, dbe_dx3 = dbe_dx[...,
                                           0], dbe_dx[..., 1], dbe_dx[..., 2]
        dbe_dx1, dbe_dx2, dbe_dx3 = dbe_dx1.reshape(
            -1, 1), dbe_dx2.reshape(-1, 1), dbe_dx3.reshape(-1, 1)

        dbe_dy = np.gradient(Be_gauss, axis=0)
        dbe_dy1, dbe_dy2, dbe_dy3 = dbe_dy[...,
                                           0], dbe_dy[..., 1], dbe_dy[..., 2]
        dbe_dy1, dbe_dy2, dbe_dy3 = dbe_dy1.reshape(
            -1, 1), dbe_dy2.reshape(-1, 1), dbe_dy3.reshape(-1, 1)

        A1 = (be1*dbe_dx2) - (be2*dbe_dx1)
        A2 = (be1*dbe_dx3) - (be3*dbe_dx1)
        A3 = (be2*dbe_dx3) - (be3*dbe_dx2)
        A4 = (-be1*dbe_dy2) + (be2*dbe_dy1)
        A5 = (-be1*dbe_dy3) + (be3*dbe_dy1)
        A6 = (-be2*dbe_dy3) + (be3*dbe_dy2)

        A = np.hstack([A1, A2, A3, A4, A5, A6])
        # print("HERE2")
        U, S, V_t = np.linalg.svd(A, full_matrices=False)
        x = V_t.T[:, -1]
        delta = np.array(
            [[-x[2], x[5], 1], [x[1], -x[4], 0], [-x[0], x[3], 0]])
        delta = np.linalg.inv(delta)
        Be_reshaped_delta = (np.linalg.inv(GBR).T @ delta @ (Be_reshaped.transpose(2,
                                                                                   0, 1).reshape(3, -1))).reshape(C, H, W).transpose(1, 2, 0)

        breakpoint()
        albedo = np.linalg.norm(Be_reshaped_delta, axis=2).reshape((H, W, 1))
        normals = Be_reshaped_delta / (albedo + 1e-8)
        derivatives = Be_reshaped_delta[..., :2] / - \
            (Be_reshaped_delta[..., 2].reshape((H, W, 1)) + 1e-8)
        depth = integrate_poisson(derivatives[..., 0], derivatives[..., 1])
        depth = (depth - depth.min())/(depth.max() - depth.min())
    else:
        albedo = np.linalg.norm(Be_reshaped, axis=2).reshape((H, W, 1))
        normals = Be_reshaped / (albedo + 1e-8)
        depth = None

    return normals, albedo, depth


def calibrated_photometric_stereo(luminance_matrix, dims):
    H, W, C = dims
    Le = load_sources()
    Be = np.linalg.inv(Le.T @ Le) @ (Le.T @ luminance_matrix)
    Be_reshaped = Be.reshape((C, H, W)).transpose(1, 2, 0)
    albedo = np.linalg.norm(Be_reshaped, axis=2).reshape((H, W, 1))
    normals = Be_reshaped / albedo
    derivatives = Be_reshaped[..., :2] / - \
        (Be_reshaped[..., 2].reshape((H, W, 1)) + 1e-8)
    depth = integrate_frankot(derivatives[..., 0], derivatives[..., 1])
    depth = (depth - depth.min())/(depth.max() - depth.min())

    return normals, albedo, None


def capture(num_captures, data_dir=Path("../lintroller"), lighting_direction="MIDDLE-CENTER"):
    out = data_dir/lighting_direction
    if not out.exists():
        out.mkdir(parents=True, exist_ok=False)

    for i in range(num_captures):
        print(f"Capturing Image {i}...")
        cmd = [
            "sudo", "gphoto2", "--set-config-value",
            f"/main/capturesettings/shutterspeed=2"
        ]
        subprocess.run(cmd)
        cmd = [
            "sudo", "gphoto2", "--capture-image-and-download", "--filename",
            out/f"{i}.%C"
        ]
        subprocess.run(cmd)


def process_raw(src_dir: Path):
    """Uses dcraw to process raw files to linear, 16-bit TIFFs.

    Args:
        src_dir: Path to directory of raw files.

    Returns:
        Writes linear, 16-bit TIFFs to src_dir
    """
    raw_imgs = list(src_dir.rglob("*.nef"))
    for raw_img in raw_imgs:
        print(f"Processing {raw_img}...")
        cmd = [
            "sudo", "dcraw", "-w", "-o", "1", "-q", "3", "-T", "-4", "-H", "0",
            str(raw_img)
        ]
        subprocess.run(cmd)


def main(args):
    data_dir = Path(args.data_dir)
    mask_available = args.mask
    image_paths = list(data_dir.rglob("*.tif"))
    if len(image_paths) == 0:
        image_paths = list(data_dir.rglob("*.tiff"))
    im0 = imageio.imread(image_paths[0])
    im0 = preprocess(im0)
    H, W, C = im0.shape
    mask_path = (data_dir / "0.png") if mask_available else None
    mydata = mask_available
    luminance_matrix = load_images(
        data_dir, mask_path=mask_path, mydata=mydata)

    for sigma in tqdm(range(1, 21)):
        if args.calibrated:
            normals, albedo, depth = calibrated_photometric_stereo(
                luminance_matrix, dims=(H, W, C))
        else:
            normals, albedo, depth = uncalibrated_photometric_stereo(
                luminance_matrix, dims=(H, W, C), enforce_integrability=args.enforce, sigma=sigma, GBR=[0, 0, 1], Q=None)
        if args.visualize:
            out = "calibrated_stereo" if args.calibrated else "uncalibrated_stereo"
            visualize(normals=normals, albedo=albedo, depth=depth,
                      sigma=sigma, GBR=[0, 0, 1], out_dir=data_dir / out, plot=False)
            if args.enforce:
                plot_surface(-depth)
    # capture(1, data_dir=data_dir, lighting_direction="BOTTOM-CENTER")
    # process_raw(Path(data_dir))


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument(
        "--mask", action=ap.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--calibrated", action=ap.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--enforce", action=ap.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--visualize", action=ap.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    main(args)
