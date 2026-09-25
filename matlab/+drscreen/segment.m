function s1 = segment(image, mask, models, original, raw)
%SEGMENT  Stage 1: optic disc, fovea, vessels, lesions, ETDRS quadrants.
%   s1 = drscreen.segment(image, mask, models)
%   s1 = drscreen.segment(image, mask, models, original, raw)
%   image    : 512x512x3 uint8 working frame from drscreen.gate (enhanced when usable)
%   mask     : 512x512 logical FOV
%   models   : struct from drscreen.loadModels; if models.unet is non-empty the
%              lesion U-Net is used (method 'unet'), else the classical
%              detectors (method 'classical'). Mirrors stage1_segment.py.
%   original : the un-enhanced 512 frame the U-Net sees (default: image),
%              colour-normalised to the DDR statistics first
%   raw      : the uploaded image as read; when models.unetHires is present its
%              classes (config/lesion_thresholds_1024.json "serves") are read at
%              the larger frame from a fresh FOV normalisation of `raw`
%
%   Output fields: method, opticDisc (centre [x y], radius), fovea (centre,
%   confidence), vessels (fraction), lesions.(MA|HE|EX|SE) (count,
%   areaFraction, centroids Nx2), hemorrhagesPerQuadrant [4], masks (struct
%   of logical 512x512), elapsedMs.

    t0 = tic;
    if nargin < 4 || isempty(original), original = image; end
    if nargin < 5, raw = []; end
    fovArea = max(nnz(mask), 1);
    od = opticDisc(original, mask);                  % un-enhanced frame (see opticDisc)
    fov = fovea(image, mask, od);
    ves = vessels(image, mask);
    if isfield(models, 'unet') && ~isempty(models.unet)
        method = 'unet';
        L = unetLesions(original, mask, od.mask, models);
        if isfield(models, 'unetHires') && ~isempty(models.unetHires) && ~isempty(raw)
            L = hiresLesions(L, raw, od.mask, models);
            method = sprintf('unet (%s at %d px)', strjoin(models.unetHiresServes, '/'), models.unetHiresSize);
        end
    else
        method = 'classical';
        [L.MA, L.HE] = redLesions(image, mask, ves.mask, od.mask);
        [L.EX, L.SE] = brightLesions(image, mask, od.mask, ves.mask);
    end
    keys = {'MA', 'HE', 'EX', 'SE'};
    lesions = struct(); masks = struct('opticDisc', od.mask, 'vessels', ves.mask);
    for i = 1:numel(keys)
        k = keys{i};
        cc = bwconncomp(L.(k));
        props = regionprops(cc, 'Centroid');
        centroids = reshape([props.Centroid], 2, []).';
        lesions.(k) = struct('count', cc.NumObjects, 'areaFraction', nnz(L.(k)) / fovArea, 'centroids', centroids);
        masks.(k) = L.(k);
    end
    quadrants = quadrantCounts(lesions.HE.centroids, fov.centre);
    s1 = struct('method', method, 'opticDisc', struct('centre', od.centre, 'radius', od.radius), ...
        'fovea', fov, 'vessels', struct('fraction', ves.fraction), 'lesions', lesions, ...
        'hemorrhagesPerQuadrant', quadrants, 'neovascularization', struct('segmented', false, ...
        'note', 'not segmented; Stage 2 reports P(NV) from the grader as PDR evidence'), ...
        'masks', masks, 'elapsedMs', round(1000 * toc(t0)));
end

% ---------------------------------------------------------------- optic disc
function od = opticDisc(image, mask)
%OPTICDISC  Brightest chromatic blob after flat-fielding (stage1_segment.optic_disc).
%   `image` is the UN-enhanced frame: enhancement divides out illumination at
%   ~17 px, smaller than a disc. Flat-field at sigma = FOV/4 removes vignetting
%   instead, and grey/white pixels (image labels, text, glare) are excluded.
%   Measured against clinician fovea marks on 1,008 MESSIDOR images:
%   backend/config/experiments/landmark_check.json.
    rgb = double(image);
    top = max(rgb, [], 3); bottom = min(rgb, [], 3);
    achromatic = (top - bottom) ./ max(top, 1) < 0.12 & top > 60 & mask;
    achromatic = imdilate(achromatic, strel('disk', 6));
    lum = 0.5 * rgb(:, :, 2) + 0.5 * rgb(:, :, 1);
    lum(achromatic) = 0;
    inside = double(mask);
    fovD = 2 * sqrt(max(nnz(mask), 1) / pi);
    illumination = imgaussfilt(lum .* inside, fovD / 4) ./ max(imgaussfilt(inside, fovD / 4), 1e-3);
    lum = lum ./ max(illumination, 8);
    smooth = imgaussfilt(lum, 12);
    inner = imerode(mask, strel('disk', 10));
    smooth(~inner) = -1;
    smooth(achromatic) = -1;
    [~, idx] = max(smooth(:));
    [cy, cx] = ind2sub(size(smooth), idx);
    fovDiameter = 2 * sqrt(nnz(mask) / pi);
    radius = fovDiameter / 13;
    % Local bright-region refinement.
    [yy, xx] = ndgrid(1:size(mask, 1), 1:size(mask, 2));
    local = hypot(xx - cx, yy - cy) <= 2.2 * radius;
    gray = double(rgb2gray(image));
    gray(achromatic) = 0;
    thresh = prctile(gray(local & mask), 90);
    bright = imclose((gray >= thresh) & local, strel('disk', 4));
    cc = bwconncomp(bright);
    if cc.NumObjects > 0
        props = regionprops(cc, 'Area', 'Centroid', 'EquivDiameter');
        [~, j] = max([props.Area]);
        if props(j).Area > pi * (0.4 * radius)^2
            fr = props(j).EquivDiameter / 2;
            if fr > 0.6 * radius && fr < 1.8 * radius
                cx = props(j).Centroid(1); cy = props(j).Centroid(2); radius = fr;
            end
        end
    end
    od.centre = [round(cx), round(cy)];
    od.radius = radius;
    od.mask = hypot(xx - cx, yy - cy) <= radius;
end

% -------------------------------------------------------------------- fovea
function f = fovea(image, mask, od)
    cx = od.centre(1); cy = od.centre(2); dd = 2 * od.radius;
    smooth = imgaussfilt(double(image(:, :, 2)), 10);
    inner = imerode(mask, strel('disk', max(round(dd), 20)));
    [H, W] = size(mask);
    [yy, xx] = ndgrid(1:H, 1:W);
    dist = hypot(xx - cx, yy - cy);
    band = dist > 1.8 * dd & dist < 3.2 * dd & abs(yy - cy) < 0.9 * dd & inner;
    props = regionprops(mask, 'Centroid');
    mc = props(1).Centroid;
    smooth = smooth + 0.12 * hypot(xx - mc(1), yy - mc(2));
    if ~any(band(:))
        direction = 1; if cx >= W / 2, direction = -1; end
        f = struct('centre', [round(cx + direction * 2.5 * dd), round(cy)], 'confidence', 0, 'method', 'geometric prior');
        return;
    end
    cand = smooth; cand(~band) = inf;
    [~, idx] = min(cand(:));
    [fy, fx] = ind2sub(size(cand), idx);
    contrast = median(smooth(band)) - smooth(fy, fx);
    f = struct('centre', [fx, fy], 'confidence', min(max(contrast / 12, 0), 1), ...
        'method', 'darkest blob in a 2-3 DD horizontal annulus');
end

% ------------------------------------------------------------------ vessels
function v = vessels(image, mask)
    green = adapthisteq(image(:, :, 2), 'ClipLimit', 0.01, 'NumTiles', [8 8]);
    inner = imerode(mask, strel('disk', 6));
    % fibermetric: dark ridges (vessels) at thicknesses 2-9 px.
    response = fibermetric(green, 2:2:9, 'ObjectPolarity', 'dark', 'StructureSensitivity', 15);
    response(~inner) = 0;
    thresh = prctile(response(inner), 88);
    bw = response >= max(thresh, 1e-6) & inner;
    bw = imopen(bw, strel('disk', 1));
    bw = bwareaopen(bw, 30);
    v = struct('mask', bw, 'fraction', nnz(bw) / max(nnz(mask), 1));
end

% -------------------------------------------------------- lesion detectors
function t = robustThreshold(values, k, floorValue)
    med = median(values);
    madv = 1.4826 * median(abs(values - med));
    t = max(med + k * max(madv, 1), floorValue);
end

function e = elongation(props)
    e = props.MajorAxisLength / max(props.MinorAxisLength, 1e-6);
end

function [MA, HE] = redLesions(image, mask, vesselMask, odMask)
    green = medfilt2(adapthisteq(image(:, :, 2), 'ClipLimit', 0.0125, 'NumTiles', [8 8]), [3 3]);
    inner = imerode(mask, strel('disk', 10));
    bh = imbothat(green, strel('disk', 9));
    bh(~inner) = 0;
    thresh = robustThreshold(double(bh(inner)), 8.0, 14.0);
    cand = double(bh) >= thresh & inner;
    cand(imdilate(vesselMask, strel('disk', 4))) = false;
    cand(imdilate(odMask, strel('disk', 8))) = false;
    cand = imopen(cand, strel('disk', 1));
    vesselNear = imdilate(vesselMask, strel('disk', 5));
    cc = bwconncomp(cand);
    props = regionprops(cc, 'Area', 'MajorAxisLength', 'MinorAxisLength', 'PixelIdxList');
    MA = false(size(mask)); HE = false(size(mask));
    for i = 1:numel(props)
        if props(i).Area < 4, continue; end
        e = elongation(props(i));
        overlap = mean(vesselNear(props(i).PixelIdxList));
        if props(i).Area < 40 && e <= 2.2
            MA(props(i).PixelIdxList) = true;
        elseif props(i).Area >= 15 && e <= 3.5 && overlap <= 0.35
            HE(props(i).PixelIdxList) = true;
        end
    end
end

function [EX, SE] = brightLesions(image, mask, odMask, vesselMask)
    green = adapthisteq(image(:, :, 2), 'ClipLimit', 0.01, 'NumTiles', [8 8]);
    inner = imerode(mask, strel('disk', 22));
    th = imtophat(green, strel('disk', 9));
    th(~inner) = 0;
    odWide = imdilate(odMask, strel('disk', 14));
    thresh = robustThreshold(double(th(inner & ~odWide)), 4.5, 18.0);
    cand = double(th) >= thresh & inner;
    cand(odWide) = false;
    cand(imdilate(vesselMask, strel('disk', 3))) = false;
    cand = imopen(cand, strel('disk', 1));
    lab = rgb2lab(image);
    bstar = lab(:, :, 3);
    yellowness = bstar - imgaussfilt(bstar, 15);
    gx = imfilter(double(green), fspecial('sobel')', 'replicate');
    gy = imfilter(double(green), fspecial('sobel'), 'replicate');
    edges = hypot(gx, gy);
    cc = bwconncomp(cand);
    props = regionprops(cc, 'Area', 'MajorAxisLength', 'MinorAxisLength', 'PixelIdxList');
    EX = false(size(mask)); SE = false(size(mask));
    for i = 1:numel(props)
        if props(i).Area < 5 || elongation(props(i)) > 3.0, continue; end
        region = false(size(mask)); region(props(i).PixelIdxList) = true;
        if mean(yellowness(region)) < 2.5, continue; end
        boundary = imdilate(region, strel('disk', 1)) & ~region;
        edgeStrength = mean(edges(boundary));
        if props(i).Area >= 60 && edgeStrength < 22
            SE(props(i).PixelIdxList) = true;
        else
            EX(props(i).PixelIdxList) = true;
        end
    end
end

function L = unetLesions(image, mask, odMask, models)
    rgb = drscreen.colourNormalise(image, mask);
    probs = drscreen.predictNet(models.unet, rgb);   % 512x512x4, MA HE EX SE
    inner = imerode(mask, strel('disk', 6));
    odWide = imdilate(odMask, strel('disk', 6));
    keys = {'MA', 'HE', 'EX', 'SE'}; minArea = [4 15 5 60];
    for i = 1:4
        bw = probs(:, :, i) >= models.unetThresholds.(keys{i}) & inner;
        if i >= 3, bw(odWide) = false; end
        L.(keys{i}) = bwareaopen(bw, minArea(i));
    end
end

function L = hiresLesions(L, raw, odMask, models)
%HIRESLESIONS  Classes served by the larger-frame network, thresholded and
%   cleaned at its own resolution (a 1-2 px microaneurysm would not survive
%   averaging to 512), then brought back to the working frame so that any
%   blob keeps at least one pixel. Mirrors the hires branch of
%   stage1_segment.unet_lesions.
    n = models.unetHiresSize;
    if size(raw, 3) == 1, raw = repmat(raw, [1 1 3]); end
    [hiImage, hiMask] = drscreen.normaliseFov(raw, n);
    rgb = drscreen.colourNormalise(hiImage, hiMask);
    probs = drscreen.predictNet(models.unetHires, rgb);
    scale = n / size(odMask, 1);
    inner = imerode(hiMask, strel('disk', round(6 * scale)));
    odWide = imdilate(imresize(odMask, [n n], 'nearest'), strel('disk', round(6 * scale)));
    keys = {'MA', 'HE', 'EX', 'SE'}; minArea = [4 15 5 60];
    minAreaHires = struct('MA', 6);      % 10th percentile of ground-truth MA area at 1024 px (DDR valid)
    for i = 1:4
        k = keys{i};
        if ~any(strcmp(models.unetHiresServes, k)), continue; end
        bw = probs(:, :, i) >= models.unetHiresThresholds.(k) & inner;
        if i >= 3, bw(odWide) = false; end
        if isfield(minAreaHires, k), m = minAreaHires.(k); else, m = minArea(i) * scale * scale; end
        bw = bwareaopen(bw, m);
        L.(k) = imresize(double(bw), 1 / scale, 'box') > 0;
    end
end

% ---------------------------------------------------------------- quadrants
function counts = quadrantCounts(centroids, foveaCentre)
    counts = zeros(1, 4);
    for i = 1:size(centroids, 1)
        dx = centroids(i, 1) - foveaCentre(1); dy = centroids(i, 2) - foveaCentre(2);
        if abs(dx) >= abs(dy)
            if dx >= 0, counts(2) = counts(2) + 1; else, counts(4) = counts(4) + 1; end
        else
            if dy >= 0, counts(3) = counts(3) + 1; else, counts(1) = counts(1) + 1; end
        end
    end
end
