function cnn = gradeCNN(s0, models, point, tta)
%GRADECNN  Stage 2 CNN grader: ordinal head -> grade, calibrated P(referable).
%   cnn = drscreen.gradeCNN(s0, models, point, tta)
%   s0     : Stage 0 struct (uses s0.original, the un-enhanced 512 frame the
%            v2 grader was trained on)
%   models : struct from drscreen.loadModels (models.grader is a dlnetwork
%            imported from the exported TensorFlow model; 512x512x3 in,
%            4 sigmoid outputs P(grade >= k) out)
%   point  : struct from drscreen.operatingPoint (Platt a, b)
%   tta    : logical; four rotations + one flip averaged (off on CPU)
%
%   Fields: grade, gradeLabel, ordinalThresholds [4], gradeProbabilities [5],
%   referableRaw, referableProbability (Platt-calibrated), nvProbability
%   (P(grade >= 4), classifier not localised), input (single 512x512x3).

    if nargin < 4, tta = false; end
    x = single(s0.original);
    if tta
        views = cat(4, x, rot90(x, 1), rot90(x, 2), rot90(x, 3), fliplr(x));
    else
        views = x;
    end
    out = predict(models.grader, views);
    if ~isa(out, 'double') && ~isa(out, 'single'), out = extractdata(out); end
    ordinal = double(mean(reshape(out, 4, []), 2)).';
    grade = sum(ordinal >= 0.5);
    labels = drscreen.icdrLabels();
    raw = ordinal(2);
    p = drscreen.calibrate(raw, point);
    cum = [1, cummin(ordinal), 0];
    perGrade = max(cum(1:end-1) - cum(2:end), 0);
    perGrade = perGrade / max(sum(perGrade), 1e-6);
    cnn = struct('grade', grade, 'gradeLabel', labels{grade + 1}, 'ordinalThresholds', ordinal, ...
        'gradeProbabilities', perGrade, 'referableRaw', raw, 'referableProbability', p, ...
        'nvProbability', ordinal(4), 'tta', tta, 'grader', point.graderTag, 'input', x);
end
