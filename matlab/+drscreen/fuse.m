function fusion = fuse(cnn, rule, point)
%FUSE  Combine the CNN grade with the rule grade; disagreement is an output.
%   fusion = drscreen.fuse(cnn, rule, point)
%   cnn   : struct from drscreen.gradeCNN (grade, gradeLabel, referableProbability, ...)
%   rule  : struct from drscreen.gradeRule
%   point : struct from drscreen.operatingPoint (thresholds.referable, thresholds.abstainBand)
%
%   The CNN grade is primary. The case is flagged for human review when the
%   grades differ by >= 2 levels, when the CNN calls referable but no lesion
%   was counted, or when P(referable) lies inside the abstain band around the
%   locked threshold. The attention-agreement flag is added by screenImage.

    threshold = point.thresholds.referable;
    band = point.thresholds.abstainBand;
    p = cnn.referableProbability;
    referable = p >= threshold;
    abstain = abs(p - threshold) <= band;
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
        reasons{end+1} = sprintf('P(referable) %.2f is within +/-%.2f of the threshold %.2f', p, band, threshold);
    end

    fusion = struct('grade', cnn.grade, 'gradeLabel', cnn.gradeLabel, 'referable', referable, ...
        'pReferable', p, 'threshold', threshold, 'abstain', abstain, 'abstainBand', band, ...
        'gradesAgreeWithinOne', disagreement <= 1, 'disagreementLevels', disagreement, ...
        'flagForReview', ~isempty(reasons), 'flagReasons', {reasons}, ...
        'agreementText', sprintf('CNN grade %d, rule grade %d', cnn.grade, rule.grade), ...
        'confidenceText', sprintf('Referable: %.2f (calibrated, threshold %.2f). Agreement: CNN grade %d, rule grade %d', ...
            p, threshold, cnn.grade, rule.grade));
end
