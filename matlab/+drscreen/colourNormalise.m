function out = colourNormalise(image, mask, referencePath)
%COLOURNORMALISE  Match the FOV's LAB mean/std to the U-Net's training statistics.
%   out = drscreen.colourNormalise(image, mask) mirrors
%   stage1_segment.colour_normalise: per-channel LAB mean and std inside the
%   FOV are mapped onto the DDR reference in config/unet_colour_reference.json,
%   so a very red or very pale retina from another camera does not produce
%   spurious hemorrhages. Applied to the UN-enhanced frame, because the
%   network was trained on raw frames. Outside the FOV the output is black.
%
%   The reference file holds OpenCV 8-bit LAB (L in 0..255, a/b offset by
%   128); MATLAB's rgb2lab gives L in 0..100 and signed a/b, so the reference
%   is converted before use. Without the file the image is returned as is.

    if nargin < 3
        referencePath = fullfile(drscreen.repoRoot(), 'backend', 'config', 'unet_colour_reference.json');
    end
    if ~isfile(referencePath)
        out = image;
        return;
    end
    ref = jsondecode(fileread(referencePath));
    refMean = [ref.lab_mean(1) * 100 / 255, ref.lab_mean(2) - 128, ref.lab_mean(3) - 128];
    refStd = [ref.lab_std(1) * 100 / 255, ref.lab_std(2), ref.lab_std(3)];
    lab = rgb2lab(image);
    inside = logical(mask);
    for c = 1:3
        ch = lab(:, :, c);
        mu = mean(ch(inside)); sd = max(std(ch(inside)), 1e-3);
        lab(:, :, c) = (ch - mu) / sd * refStd(c) + refMean(c);
    end
    out = uint8(255 * min(max(lab2rgb(lab), 0), 1));
    out(repmat(~inside, [1 1 3])) = 0;
end
