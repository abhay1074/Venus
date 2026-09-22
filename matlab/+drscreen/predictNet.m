function y = predictNet(net, x)
%PREDICTNET  Run one image through an imported or native dlnetwork.
%   y = drscreen.predictNet(net, x)  x: H x W x C single (0-255 as trained).
%   Networks imported with importNetworkFromTensorFlow from a Keras 3
%   SavedModel (no keras_metadata.pb) arrive as one opaque call layer behind
%   an InputLayer that takes an UNFORMATTED batch-first array (1 x H x W x C,
%   format 'UUUU'); a native dlnetwork (drscreen.buildUnet) has an
%   imageInputLayer and takes a formatted 'SSC' image. Verified in R2026a:
%   gate / quality / grader outputs match Python to <= 2e-6 this way.
%   Returns a plain single array with the batch dimension removed.
    x = single(x);
    if isa(net.Layers(end), 'nnet.cnn.layer.InputLayer') || any(arrayfun(@(l) isa(l, 'nnet.cnn.layer.InputLayer'), net.Layers))
        out = predict(net, dlarray(reshape(x, [1 size(x)]), 'UUUU'));
        y = extractdata(out);
        y = reshape(y, [size(y, 2:ndims(y)) 1]);          % drop the leading batch dim
        if isvector(y), y = y(:).'; end
    else
        out = predict(net, dlarray(x, 'SSC'));
        y = extractdata(out);
    end
    y = single(y);
end
