function out = enhance(image, mask)
%ENHANCE  Adaptive enhancement for 'usable' images (Stage 0).
%   out = drscreen.enhance(image, mask)
%   1. Illumination normalisation: divide by a normalised-convolution Gaussian
%      estimate of the illumination (sigma = width/30, FOV-only so the black
%      surround does not distort the rim) and rescale to a mid-grey target.
%      Colour and local contrast are preserved for the lesion detectors.
%   2. Bilateral filter with a small spatial sigma so microaneurysms survive.
%   Never bleeds outside the FOV.

    sigma = size(image, 2) / 30;
    gray = double(rgb2gray(image));
    weight = double(mask);
    illumination = imgaussfilt(gray .* weight, sigma) ./ max(imgaussfilt(weight, sigma), 1e-3);
    illumination(~mask) = 1;
    target = max(prctile(gray(mask), 60), 1);
    target = min(max(max(target, 90), 90), 140);
    gain = min(max(target ./ max(illumination, 8), 0.6), 4.0);
    corrected = uint8(min(max(double(image) .* gain, 0), 255));
    out = imbilatfilt(corrected, 25^2, 3);
    out(repmat(~mask, [1 1 3])) = 0;
end
