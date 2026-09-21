function fusion = fuse(cnn, rule, point, policy)
%FUSE  Combine the CNN grade with the rule grade; disagreement is an output.
%   fusion = drscreen.fuse(cnn, rule, point)
%   fusion = drscreen.fuse(cnn, rule, point, policy)
%   cnn    : struct from drscreen.gradeCNN (grade, gradeLabel, referableProbability, ...)
%   rule   : struct from drscreen.gradeRule
%   point  : struct from drscreen.operatingPoint (thresholds.referable, thresholds.abstainBand)
%   policy : struct from drscreen.reviewPolicy (default: read from config)
%
%   The CNN grade is primary. The case is flagged for human review when the
%   grades differ by >= 2 levels, when the CNN calls referable but no lesion
%   was counted, or when P(referable) lies inside the abstain band around the
%   locked threshold. The band is +/- policy.abstainBandLogit in logit space
%   when the validation-chosen policy exists (symmetric around a threshold
%   near 0.1, where +/-0.05 in probability is not), else the architecture's
%   +/-0.05 in probability. The attention-agreement flag is added by
%   screenImage. Mirrors backend.venus.stage2_grade.fuse.

    if nargin < 4, policy = drscreen.reviewPolicy(); end
    threshold = point.thresholds.referable;
    p = cnn.referableProbability;
    referable = p >= threshold;
    if ~isnan(policy.abstainBandLogit)
        half = policy.abstainBandLogit;
        lo = sigmoid(logit(threshold) - half);
        hi = sigmoid(logit(threshold) + half);
        abstain = p >= lo && p <= hi;
        bandText = sprintf('+/-%.2f logit (%.3f-%.3f)', half, lo, hi);
    else
        lo = threshold - point.thresholds.abstainBand;
        hi = threshold + point.thresholds.abstainBand;
        abstain = abs(p - threshold) <= point.thresholds.abstainBand;
        bandText = sprintf('+/-%.2f', point.thresholds.abstainBand);
    end
    band = hi - threshold;
    disagreement = abs(cnn.grade - rule.grade);
    totalLesions = rule.counts.MA + rule.counts.HE + rule.counts.EX + rule.counts.SE;

    reasons = {};
    if disagreement >= 2
        reasons{end+1} = sprintf('CNN grade %d and rule grade %d differ by %d levels', cnn.grade, rule.grade, disagreement);
    end
    if referable && totalLesions == 0
        reasons{end+1} = 'CNN calls referable but no lesion was found';
    end
    if abstain
        reasons{end+1} = sprintf('P(referable) %.2f is within the abstain band %s around the threshold %.2f', p, bandText, threshold);
    end

    fusion = struct('grade', cnn.grade, 'gradeLabel', cnn.gradeLabel, 'referable', referable, ...
        'pReferable', p, 'threshold', threshold, 'abstain', abstain, ...
        'abstainBand', round(band * 10000) / 10000, 'abstainLow', round(lo * 10000) / 10000, ...
        'abstainHigh', round(hi * 10000) / 10000, 'reviewPolicy', policy.chosenOn, ...
        'gradesAgreeWithinOne', disagreement <= 1, 'disagreementLevels', disagreement, ...
        'flagForReview', ~isempty(reasons), 'flagReasons', {reasons}, ...
        'agreementText', sprintf('CNN grade %d, rule grade %d', cnn.grade, rule.grade), ...
        'confidenceText', sprintf('Referable: %.2f (calibrated, threshold %.2f). Agreement: CNN grade %d, rule grade %d', ...
            p, threshold, cnn.grade, rule.grade));
end

function y = logit(x)
    x = min(max(x, 1e-6), 1 - 1e-6);
    y = log(x ./ (1 - x));
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end
