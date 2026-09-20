function f = qualityFeatures(image, mask, geometry)
%QUALITYFEATURES  Handcrafted image-quality features inside the FOV.
%   f = drscreen.qualityFeatures(image, mask, geometry) -> struct with
%   sharpness (variance of Laplacian on green), illuminationUniformity
%   (darkest periphery cell / centre mean on a 5x5 grid), saturatedFraction,
%   darkFraction, fovCoverage, fovCircularity. Mirrors stage0_gate.py.

    green = double(image(:, :, 2));
    gray = double(rgb2gray(image));
    lap = imfilter(green, fspecial('laplacian', 0), 'replicate');
    f.sharpness = var(lap(mask));

    [H, W] = size(gray);
    ch = floor(H / 5); cw = floor(W / 5);
    centre = []; periphery = [];
    for r = 0:4
        for c = 0:4
            block = gray(r * ch + 1:(r + 1) * ch, c * cw + 1:(c + 1) * cw);
            bm = mask(r * ch + 1:(r + 1) * ch, c * cw + 1:(c + 1) * cw);
            if mean(bm(:)) > 0.6
                v = mean(block(bm));
                if r >= 1 && r <= 3 && c >= 1 && c <= 3, centre(end+1) = v; else, periphery(end+1) = v; end %#ok<AGROW>
            end
        end
    end
    if ~isempty(centre) && ~isempty(periphery)
        f.illuminationUniformity = min(max(min(periphery) / max(mean(centre), 1e-6), 0), 1.5);
    else
        f.illuminationUniformity = 1.0;
    end
    f.saturatedFraction = mean(gray(mask) >= 245);
    f.darkFraction = mean(gray(mask) <= 12);
    f.fovCoverage = geometry.coverage;
    f.fovCircularity = geometry.circularity;
end
