function models = loadModels(modelDir)
%LOADMODELS  Import the exported networks and the locked operating point.
%   models = drscreen.loadModels() looks in <repo>/models/export for the
%   TensorFlow SavedModel folders written by `python -m backend.eval.export_models`:
%       grader_v2/     EfficientNet-B3 at 512, 4 sigmoid outputs (P(grade >= k))
%       modality_gate/ EfficientNet-B0 at 224, 5-way softmax
%       quality_cnn/   EfficientNet-B0 at 256, 3-way softmax (optional)
%       lesion_unet/   U-Net at 512, 4 sigmoid channels (optional)
%   and imports each with importNetworkFromTensorFlow (falling back to the
%   ONNX file of the same name with importNetworkFromONNX). Imported networks
%   are cached as .mat next to the export so later loads take a second.
%
%   Fields: grader, gate, quality ([] if absent), unet ([] if absent),
%   unetThresholds (from config/lesion_thresholds.json), unetHires ([] if
%   absent; the 1024 px network with unetHiresThresholds / unetHiresServes /
%   unetHiresSize from config/lesion_thresholds_1024.json), point, policy.

    if nargin < 1, modelDir = fullfile(drscreen.repoRoot(), 'models', 'export'); end
    models.point = drscreen.operatingPoint();
    models.policy = drscreen.reviewPolicy();
    models.grader = importOne(fullfile(modelDir, 'grader_v2'), true);
    % Grad-CAM: the importer folds the grader into one opaque layer, so the
    % feature maps come from their own export (inputs -> top_activation) and
    % the dense head's weights from grader_v2_head.json (see explain.m).
    models.graderFeatures = importOne(fullfile(modelDir, 'grader_v2_features'), false);
    headPath = fullfile(modelDir, 'grader_v2_head.json');
    models.graderHead = [];
    if isfile(headPath), models.graderHead = jsondecode(fileread(headPath)); end
    models.gate = importOne(fullfile(modelDir, 'modality_gate'), true);
    models.quality = importOne(fullfile(modelDir, 'quality_cnn'), false);
    % The U-Net is built natively from the Keras checkpoint (buildUnet: the
    % TensorFlow importer has no transposed convolution); the SavedModel is
    % the fallback when the checkpoint is absent.
    weightsDir = fullfile(drscreen.repoRoot(), 'backend', 'weights');
    if isfile(fullfile(weightsDir, 'lesion_unet.weights.h5'))
        models.unet = cachedNative(fullfile(modelDir, 'lesion_unet'), @() drscreen.buildUnet(fullfile(weightsDir, 'lesion_unet.weights.h5'), 512));
    else
        models.unet = importOne(fullfile(modelDir, 'lesion_unet'), false);
    end
    models.unetThresholds = [];
    thrPath = fullfile(drscreen.repoRoot(), 'backend', 'config', 'lesion_thresholds.json');
    if ~isempty(models.unet) && isfile(thrPath)
        models.unetThresholds = jsondecode(fileread(thrPath)).thresholds;
    else
        models.unet = [];
    end
    % Optional larger-frame lesion network for the classes its spec lists.
    models.unetHires = []; models.unetHiresThresholds = []; models.unetHiresServes = {}; models.unetHiresSize = 1024;
    hiPath = fullfile(drscreen.repoRoot(), 'backend', 'config', 'lesion_thresholds_1024.json');
    if ~isempty(models.unet) && isfile(hiPath)
        spec = jsondecode(fileread(hiPath));
        if isfile(fullfile(weightsDir, 'lesion_unet_1024.weights.h5'))
            net = cachedNative(fullfile(modelDir, 'lesion_unet_1024'), @() drscreen.buildUnet(fullfile(weightsDir, 'lesion_unet_1024.weights.h5'), spec.frame_size));
        else
            net = importOne(fullfile(modelDir, 'lesion_unet_1024'), false);
        end
        if ~isempty(net) && isfield(spec, 'serves') && ~isempty(spec.serves)
            models.unetHires = net;
            models.unetHiresThresholds = spec.thresholds;
            models.unetHiresServes = cellstr(spec.serves);
            models.unetHiresSize = spec.frame_size;
        end
    end
end

function net = cachedNative(folder, builder)
    % Build once, cache next to the export like the imported networks.
    cache = [folder '.mat'];
    if isfile(cache)
        s = load(cache, 'net'); net = s.net; return;
    end
    net = builder();
    if ~isfolder(fileparts(cache)), mkdir(fileparts(cache)); end
    save(cache, 'net');
end

function net = importOne(folder, required)
    % The importer writes a +<model> package of generated layer code into the
    % current folder, and the cached .mat needs that package on the path: both
    % live next to the export (models/export, not committed), never in matlab/.
    exportDir = fileparts(folder);
    addpath(exportDir);
    cache = [folder '.mat'];
    if isfile(cache)
        s = load(cache, 'net'); net = s.net; return;
    end
    net = [];
    if isfolder(folder)
        previous = cd(exportDir); restore = onCleanup(@() cd(previous));
        net = importNetworkFromTensorFlow(folder);
        clear restore
    elseif isfile([folder '.onnx'])
        net = importNetworkFromONNX([folder '.onnx']);
    elseif required
        error('drscreen:loadModels', 'required model not found: %s (run python -m backend.eval.export_models)', folder);
    end
    if ~isempty(net)
        save(cache, 'net');
    end
end
