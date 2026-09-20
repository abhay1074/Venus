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
%   unetThresholds (from config/lesion_thresholds.json), point.

    if nargin < 1, modelDir = fullfile(drscreen.repoRoot(), 'models', 'export'); end
    models.point = drscreen.operatingPoint();
    models.grader = importOne(fullfile(modelDir, 'grader_v2'), true);
    models.gate = importOne(fullfile(modelDir, 'modality_gate'), true);
    models.quality = importOne(fullfile(modelDir, 'quality_cnn'), false);
    models.unet = importOne(fullfile(modelDir, 'lesion_unet'), false);
    models.unetThresholds = [];
    thrPath = fullfile(drscreen.repoRoot(), 'backend', 'config', 'lesion_thresholds.json');
    if ~isempty(models.unet) && isfile(thrPath)
        models.unetThresholds = jsondecode(fileread(thrPath)).thresholds;
    else
        models.unet = [];
    end
end

function net = importOne(folder, required)
    cache = [folder '.mat'];
    if isfile(cache)
        s = load(cache, 'net'); net = s.net; return;
    end
    net = [];
    if isfolder(folder)
        net = importNetworkFromTensorFlow(folder);
    elseif isfile([folder '.onnx'])
        net = importNetworkFromONNX([folder '.onnx']);
    elseif required
        error('drscreen:loadModels', 'required model not found: %s (run python -m backend.eval.export_models)', folder);
    end
    if ~isempty(net)
        save(cache, 'net');
    end
end
