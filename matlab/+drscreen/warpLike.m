function out = warpLike(array, geometry, method)
%WARPLIKE  Apply the crop/pad/resize of normaliseFov to another array.
%   out = drscreen.warpLike(mask, geometry, 'nearest')
    if nargin < 3, method = 'nearest'; end
    b = geometry.bbox; p = geometry.pad; sz = geometry.size;
    x = b(1); y = b(2); w = b(3); h = b(4); oy = p(1); ox = p(2); side = p(3);
    crop = array(y:y + h - 1, x:x + w - 1, :);
    canvas = zeros(side, side, size(array, 3), class(array));
    canvas(oy + 1:oy + h, ox + 1:ox + w, :) = crop;
    out = imresize(canvas, [sz sz], method);
end
