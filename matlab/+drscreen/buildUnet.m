function net = buildUnet(weightsPath, imageSize, base, levels)
%BUILDUNET  The lesion U-Net as a native dlnetwork with the trained Keras weights.
%   net = drscreen.buildUnet(weightsPath) builds backend.venus.nets.lesion_unet
%   layer by layer (4 levels, 32 base filters, conv-BN-ReLU x2 per block,
%   2x2 stride-2 transposed convolutions, skip concatenations, 1x1 sigmoid
%   head) and loads the weights from the Keras .h5 checkpoint. The TensorFlow
%   importer has no Conv2DBackpropInput (transposed convolution), so the
%   SavedModel comes in uninitialised; this is the exact network instead.
%
%   Keras 3 saves the checkpoint under default layer names in creation order
%   (conv2d, conv2d_1, ..., batch_normalization_N, conv2d_transpose_N), which
%   is the order the builder creates them, so the mapping is positional:
%   encoder blocks, bottleneck, decoder blocks (deepest first), head.
%   BatchNormalization stores [gamma beta mean variance], epsilon 1e-3.
%   h5read returns HDF5 (row-major) arrays with reversed dimensions, hence
%   the permutes. Verified numerically against the Python network on the
%   sample images (see tests/CrossCheckTest / octave_smoke notes).

    if nargin < 2 || isempty(imageSize), imageSize = 512; end
    if nargin < 3 || isempty(base), base = 32; end
    if nargin < 4 || isempty(levels), levels = 4; end
    if nargin < 1 || isempty(weightsPath)
        weightsPath = fullfile(drscreen.repoRoot(), 'backend', 'weights', 'lesion_unet.weights.h5');
    end

    convIndex = 0; bnIndex = 0; upIndex = 0;
    function w = var(group, k)
        w = h5read(weightsPath, sprintf('/layers/%s/vars/%d', group, k));
    end
    function name = counted(prefix, idx)
        if idx == 0, name = prefix; else, name = sprintf('%s_%d', prefix, idx); end
    end
    function layers = convBlock(filters, tag)
        layers = [];
        for c = 1:2
            W = permute(var(counted('conv2d', convIndex), 0), [4 3 2 1]);        % kh kw cin cout
            g = var(counted('batch_normalization', bnIndex), 0); b = var(counted('batch_normalization', bnIndex), 1);
            mu = var(counted('batch_normalization', bnIndex), 2); v = var(counted('batch_normalization', bnIndex), 3);
            layers = [layers
                convolution2dLayer(3, filters, 'Padding', 'same', 'Name', sprintf('%s_c%d', tag, c), ...
                    'Weights', single(W), 'Bias', zeros(1, 1, filters, 'single'), 'BiasLearnRateFactor', 0)
                batchNormalizationLayer('Name', sprintf('%s_bn%d', tag, c), 'Epsilon', 1e-3, ...
                    'Scale', reshape(single(g), 1, 1, []), 'Offset', reshape(single(b), 1, 1, []), ...
                    'TrainedMean', reshape(single(mu), 1, 1, []), 'TrainedVariance', reshape(single(v), 1, 1, []))
                reluLayer('Name', sprintf('%s_a%d', tag, c))]; %#ok<AGROW>
            convIndex = convIndex + 1; bnIndex = bnIndex + 1;
        end
    end

    lg = layerGraph();
    lg = addLayers(lg, [imageInputLayer([imageSize imageSize 3], 'Normalization', 'none', 'Name', 'fundus')
                        functionLayer(@(x) x / 255, 'Name', 'rescale', 'Formattable', false)]);
    prev = 'rescale';
    skips = cell(1, levels);
    for level = 0:levels - 1
        tag = sprintf('enc%d', level);
        lg = addLayers(lg, [convBlock(base * 2 ^ level, tag); maxPooling2dLayer(2, 'Stride', 2, 'Name', sprintf('pool%d', level))]);
        lg = connectLayers(lg, prev, [tag '_c1']);
        skips{level + 1} = [tag '_a2'];
        prev = sprintf('pool%d', level);
    end
    lg = addLayers(lg, convBlock(base * 2 ^ levels, 'bottleneck'));
    lg = connectLayers(lg, prev, 'bottleneck_c1');
    prev = 'bottleneck_a2';
    for level = levels - 1:-1:0
        filters = base * 2 ^ level;
        Wt = permute(var(counted('conv2d_transpose', upIndex), 0), [4 3 2 1]);   % kh kw out in = [h w numFilters numChannels]
        bt = var(counted('conv2d_transpose', upIndex), 1);
        upIndex = upIndex + 1;
        up = sprintf('up%d', level); cat = sprintf('cat%d', level); tag = sprintf('dec%d', level);
        lg = addLayers(lg, transposedConv2dLayer(2, filters, 'Stride', 2, 'Name', up, 'Weights', single(Wt), 'Bias', reshape(single(bt), 1, 1, [])));
        lg = addLayers(lg, concatenationLayer(3, 2, 'Name', cat));
        lg = addLayers(lg, convBlock(filters, tag));
        lg = connectLayers(lg, prev, up);
        lg = connectLayers(lg, up, [cat '/in1']);
        lg = connectLayers(lg, skips{level + 1}, [cat '/in2']);
        lg = connectLayers(lg, cat, [tag '_c1']);
        prev = [tag '_a2'];
    end
    Wh = permute(var(counted('conv2d', convIndex), 0), [4 3 2 1]); bh = var(counted('conv2d', convIndex), 1);
    lg = addLayers(lg, [convolution2dLayer(1, 4, 'Name', 'lesions', 'Weights', single(Wh), 'Bias', reshape(single(bh), 1, 1, []))
                        sigmoidLayer('Name', 'sigmoid')]);
    lg = connectLayers(lg, prev, 'lesions');
    net = dlnetwork(lg);
end
