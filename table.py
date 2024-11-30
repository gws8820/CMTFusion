import os
import cv2
import numpy as np
import pandas as pd
from scipy.fftpack import dct
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import estimate_sigma
from skimage import util
from skimage.filters import sobel
import scipy.io
import scipy.ndimage
import scipy.special
from os.path import join  # Make sure to import join here

gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range) 
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)

def aggd_features(imdata):
    imdata = imdata.flatten()
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = np.sqrt(np.average(left_data)) if len(left_data) > 0 else 0
    right_mean_sqrt = np.sqrt(np.average(right_data)) if len(right_data) > 0 else 0
    gamma_hat = left_mean_sqrt / right_mean_sqrt if right_mean_sqrt != 0 else np.inf
    imdata2_mean = np.mean(imdata2)
    r_hat = (np.average(np.abs(imdata)) ** 2) / (np.average(imdata2)) if imdata2_mean != 0 else np.inf
    rhat_norm = r_hat * (((gamma_hat ** 3 + 1) * (gamma_hat + 1)) / (gamma_hat ** 2 + 1) ** 2)
    pos = np.argmin((prec_gammas - rhat_norm) ** 2)
    alpha = gamma_range[pos]
    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)
    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt
    N = (br - bl) * (gam2 / gam1)
    return alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt

def paired_product(new_im):
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)
    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im
    return H_img, V_img, D1_img, D2_img

def gen_gauss_window(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights

def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    assert len(image.shape) == 2
    h, w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = image.astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image ** 2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
    return (image - mu_image) / (var_image + C), var_image, mu_image

def _niqe_extract_subband_feats(mscncoefs):
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    return np.array([alpha_m, (bl + br) / 2.0, alpha1, N1, bl1, br1, alpha2, N2, bl2, br2, alpha3, N3, bl3, bl3, alpha4, N4, bl4, bl4])

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)

def extract_on_patches(img, patch_size):
    h, w = img.shape
    patch_size = int(patch_size)
    patches = [img[j:j + patch_size, i:i + patch_size] for j in range(0, h - patch_size + 1, patch_size) for i in range(0, w - patch_size + 1, patch_size)]
    patches = np.array(patches)
    patch_features = [_niqe_extract_subband_feats(p) for p in patches]
    return np.array(patch_features)

def _get_patches_generic(img, patch_size, is_train, stride):
    h, w = img.shape
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)
    hoffset = (h % patch_size)
    woffset = (w % patch_size)
    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]
    img = img.astype(np.float32)
    img2 = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)
    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)
    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size / 2)
    feats = np.hstack((feats_lvl1, feats_lvl2))
    return feats

def mutual_information(hgram):
    """ Mutual information for joint histogram """
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def gradient_magnitude(image):
    """Calculate gradient magnitude and direction."""
    grad_x = sobel(image)
    grad_y = sobel(image.T).T
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    return magnitude, direction

def calculate_qabf(fused_image, ir_image, vis_image):
    fused_mag, fused_dir = gradient_magnitude(fused_image)
    ir_mag, ir_dir = gradient_magnitude(ir_image)
    vis_mag, vis_dir = gradient_magnitude(vis_image)

    def quality_index(mag_f, mag_s, dir_f, dir_s):
        QG = (2 * mag_f * mag_s + 0.1) / (mag_f**2 + mag_s**2 + 0.1)
        QD = (2 * np.cos(dir_f - dir_s) + 0.1) / (2 + 0.1)
        return QG * QD

    Q_ir = quality_index(fused_mag, ir_mag, fused_dir, ir_dir)
    Q_vis = quality_index(fused_mag, vis_mag, fused_dir, vis_dir)

    QABF = (Q_ir + Q_vis) / 2
    return np.mean(QABF)

def calculate_fmidct(fused_image, ir_image, vis_image):
    ir_dct = dct(dct(ir_image.T, norm='ortho').T, norm='ortho')
    vis_dct = dct(dct(vis_image.T, norm='ortho').T, norm='ortho')
    fused_dct = dct(dct(fused_image.T, norm='ortho').T, norm='ortho')

    hist_2d_ir, _, _ = np.histogram2d(ir_dct.ravel(), fused_dct.ravel(), bins=20)
    hist_2d_vis, _, _ = np.histogram2d(vis_dct.ravel(), fused_dct.ravel(), bins=20)

    mi_ir = mutual_information(hist_2d_ir)
    mi_vis = mutual_information(hist_2d_vis)

    return (mi_ir + mi_vis) / 2

def calculate_fmiw(fused_image, ir_image, vis_image):
    ir_wavelet = dct(dct(ir_image.T, norm='ortho').T, norm='ortho')
    vis_wavelet = dct(dct(vis_image.T, norm='ortho').T, norm='ortho')
    fused_wavelet = dct(dct(fused_image.T, norm='ortho').T, norm='ortho')

    hist_2d_ir, _, _ = np.histogram2d(ir_wavelet.ravel(), fused_wavelet.ravel(), bins=20)
    hist_2d_vis, _, _ = np.histogram2d(vis_wavelet.ravel(), fused_wavelet.ravel(), bins=20)

    mi_ir = mutual_information(hist_2d_ir)
    mi_vis = mutual_information(hist_2d_vis)

    return (mi_ir + mi_vis) / 2

def calculate_ms_ssim(fused_image, ir_image, vis_image):
    ssim_ir = ssim(ir_image, fused_image, data_range=fused_image.max() - fused_image.min(), channel_axis=True)
    ssim_vis = ssim(vis_image, fused_image, data_range=fused_image.max() - fused_image.min(), channel_axis=True)
    return (ssim_ir + ssim_vis) / 2

def calculate_scd(fused_image, ir_image, vis_image):
    def correlation_coefficient(image1, image2):
        image1_mean = np.mean(image1)
        image2_mean = np.mean(image2)
        numerator = np.mean((image1 - image1_mean) * (image2 - image2_mean))
        denominator = np.std(image1) * np.std(image2)
        return numerator / denominator

    scd_ir = correlation_coefficient(ir_image, fused_image)
    scd_vis = correlation_coefficient(vis_image, fused_image)
    return (scd_ir + scd_vis) / 2

def calculate_brisque(fused_image):
    img = util.img_as_float(fused_image)
    sigma = estimate_sigma(img, channel_axis=True, average_sigmas=True)
    img = util.random_noise(img, var=sigma**2)
    glcm = graycomatrix((img * 255).astype('uint8'), [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    ASM = graycoprops(glcm, 'ASM').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    features = np.hstack([contrast, dissimilarity, homogeneity, ASM, energy, correlation])
    return features.mean()

def calculate_niqe(inputImgData, module_path='src/niqe'):
    patch_size = 48
    params = scipy.io.loadmat(join(module_path, 'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]
    M, N = inputImgData.shape
    assert M > (patch_size * 2 + 1), "niqe called with small frame size, requires > 97x97 resolution video using current training parameters"
    assert N > (patch_size * 2 + 1), "niqe called with small frame size, requires > 97x97 resolution video using current training parameters"
    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)
    X = sample_mu - pop_mu
    covmat = ((pop_cov + sample_cov) / 2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X.T))
    return niqe_score

# Initialize the DataFrame
columns = ["ID", "Q_AB/F", "FMI_dct", "FMI_w", "MS-SSIM", "SCD", "BRISQUE", "NIQE"]
results_df = pd.DataFrame(columns=columns)

# Sort and filter file lists
ir_files = sorted([f for f in os.listdir('dataset_inference/KAIST') if f.startswith('IR') and f.endswith('.png')])
vis_files = sorted([f for f in os.listdir('dataset_inference/KAIST') if f.startswith('VIS') and f.endswith('.png')])

# Verify that the number of IR and VIS files match
assert len(ir_files) == len(vis_files), "The number of IR and VIS files does not match."

# Data processing loop
for index, (ir_file, vis_file) in enumerate(zip(ir_files, vis_files), start=1):
    ir_image_path = os.path.join('dataset_inference/KAIST', ir_file)
    vis_image_path = os.path.join('dataset_inference/KAIST', vis_file)
    fused_image_path = f'result_images/KAIST/{index}.png'

    ir_image = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)
    vis_image = cv2.imread(vis_image_path, cv2.IMREAD_GRAYSCALE)
    fused_image = cv2.imread(fused_image_path, cv2.IMREAD_GRAYSCALE)

    # Verify that the images are loaded
    if ir_image is None or vis_image is None or fused_image is None:
        print(f"Error loading images for set {index}: IR={ir_file}, VIS={vis_file}")
        continue

    # Calculate metrics
    qabf = calculate_qabf(fused_image, ir_image, vis_image)
    fmidct = calculate_fmidct(fused_image, ir_image, vis_image)
    fmiw = calculate_fmiw(fused_image, ir_image, vis_image)
    ms_ssim_val = calculate_ms_ssim(fused_image, ir_image, vis_image)
    scd = calculate_scd(fused_image, ir_image, vis_image)
    brisque_val = calculate_brisque(fused_image)
    niqe_val = calculate_niqe(fused_image)

    # Add the results to the DataFrame
    new_row = pd.DataFrame([{
        "ID": index,
        "Q_AB/F": qabf,
        "FMI_dct": fmidct,
        "FMI_w": fmiw,
        "MS-SSIM": ms_ssim_val,
        "SCD": scd,
        "BRISQUE": brisque_val,
        "NIQE": niqe_val
    }])
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# Save the results to a CSV file
results_df.to_csv('./result_table/KAIST.csv', index=False)
print("Results saved to result_table/KAIST.csv")