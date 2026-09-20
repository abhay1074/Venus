function [image, mask, geometry] = normaliseFov(imageRGB, sz)
%NORMALISEFOV  Otsu FOV mask on red, convex hull, crop, pad square, resize.
%   [image, mask, geometry] = drscreen.normaliseFov(imageRGB, 512)
%   geometry: bbox [x y w h] (1-based), pad [oy ox side], coverage,
%   circularity, sourceSize [w h], size. Use drscreen.warpLike to apply the
%   same transform to a label mask.

    if nargin < 2, sz = 512; end
    red = imgaussfilt(imageRGB(:, :, 1), 3);
    level = graythresh(red);
    bw = imbinarize(red, level);
    if mean(bw(:)) < 0.10
        bw = red > 12;
    end
    bw = bwareafilt(bw, 1);
    bw = imclose(bw, strel('square', 15));
    bw = bwconvhull(bw);
    if ~any(bw(:))
        error('drscreen:gate', 'No field of view could be found in the image.');
    end
    props = regionprops(bw, 'BoundingBox', 'Area', 'Perimeter');
    bbox = round(props(1).BoundingBox);
    x = max(bbox(1), 1); y = max(bbox(2), 1); w = bbox(3); h = bbox(4);
    [H, W] = size(bw);
    w = min(w, W - x + 1); h = min(h, H - y + 1);
    area = props(1).Area; perimeter = props(1).Perimeter;
    circularity = min(4 * pi * area / max(perimeter, 1)^2, 1);
    coverage = area / (H * W);

    side = max(w, h);
    oy = floor((side - h) / 2); ox = floor((side - w) / 2);
    canvas = zeros(side, side, 3, 'uint8');
    canvasMask = false(side, side);
    canvas(oy + 1:oy + h, ox + 1:ox + w, :) = imageRGB(y:y + h - 1, x:x + w - 1, :);
    canvasMask(oy + 1:oy + h, ox + 1:ox + w) = bw(y:y + h - 1, x:x + w - 1);
    image = imresize(canvas, [sz sz], 'bilinear');
    mask = imresize(canvasMask, [sz sz], 'nearest');
    geometry = struct('bbox', [x y w h], 'pad', [oy ox side], 'coverage', coverage, ...
        'circularity', circularity, 'sourceSize', [W H], 'size', sz);
end
