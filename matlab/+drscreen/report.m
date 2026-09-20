function paths = report(result, outDir)
%REPORT  One-page PDF (MATLAB Report Generator) and JSON sidecar.
%   paths = drscreen.report(result, outDir) writes <session>.pdf and
%   <session>.json into outDir and returns their paths. The page: session,
%   capture time, quality, three image panels (original, lesion overlay,
%   Grad-CAM), grade evidence, confidence and review flags, priority tier,
%   recommendation, and the model version + calibration fingerprint footer.

    import mlreportgen.dom.*
    if nargin < 2, outDir = fullfile(drscreen.repoRoot(), 'backend', 'reports'); end
    if ~isfolder(outDir), mkdir(outDir); end
    pdfPath = fullfile(outDir, [result.sessionId '.pdf']);
    d = Document(fullfile(outDir, result.sessionId), 'pdf');
    d.PageLayout = PDFPageLayout(); d.PageLayout.PageMargins.Left = '14mm'; d.PageLayout.PageMargins.Right = '14mm';
    open(d);

    h = Heading1('Venus AI - Diabetic Retinopathy Screening Report'); h.Color = '#102A43'; append(d, h);
    q = result.stage0.quality;
    sub = Paragraph(sprintf('Session %s   |   captured %s   |   quality: %s (score %.2f%s)', result.sessionId, ...
        result.capturedAt, q.label, q.score, ifelse(q.enhanced, ', enhanced', '')));
    sub.Color = '#64748B'; sub.FontSize = '9pt'; append(d, sub);

    f = result.stage2.fusion; t = result.stage5;
    verdict = ifelse(f.referable, 'REFERABLE DR', 'NOT REFERABLE');
    if f.flagForReview, verdict = [verdict '  -  HUMAN REVIEW']; end
    v = Paragraph(verdict); v.Bold = true; v.FontSize = '20pt';
    v.Color = ifelse(f.flagForReview, '#B45309', ifelse(f.referable, '#BE123C', '#0F766E')); append(d, v);
    labels = drscreen.icdrLabels();
    append(d, Paragraph(sprintf('ICDR grade %d - %s      P(referable) = %.2f (calibrated; threshold %.2f)%s', ...
        f.grade, labels{f.grade + 1}, f.pReferable, f.threshold, ifelse(f.abstain, '   ABSTAIN BAND', ''))));
    append(d, Paragraph(sprintf('Priority tier %s - %s: %s, %s', t.tier, t.label, t.deadlineText, t.facility)));

    % Image panels
    tmp = tempname; mkdir(tmp);
    panels = {'original', 'Original'; 'lesions', 'Lesion overlay'; 'gradcamReferable', 'Grad-CAM (referable head)'};
    tbl = Table(); tbl.Width = '100%';
    row = TableRow(); cap = TableRow();
    for i = 1:size(panels, 1)
        file = fullfile(tmp, [panels{i, 1} '.png']);
        imwrite(result.stage3.overlays.(panels{i, 1}), file);
        img = Image(file); img.Width = '58mm'; img.Height = '58mm';
        append(row, TableEntry(img));
        c = Paragraph(panels{i, 2}); c.FontSize = '8pt'; c.Color = '#64748B'; append(cap, TableEntry(c));
    end
    append(tbl, row); append(tbl, cap); append(d, tbl);

    % Evidence
    r = result.stage2.rule; c = result.stage2.cnn; s1 = result.stage1; a = result.stage3.attentionAgreement;
    append(d, headed('Grade evidence'));
    lines = {sprintf('CNN grader: grade %d (%s), five-grade probabilities %s', c.grade, c.gradeLabel, ...
        strjoin(arrayfun(@(i) sprintf('G%d %.2f', i - 1, c.gradeProbabilities(i)), 1:5, 'UniformOutput', false), ', '));
        sprintf('Rule grader: grade %d (%s)', r.grade, r.gradeLabel)};
    lines = [lines; cellfun(@(x) ['- ' x], r.trace(:), 'UniformOutput', false)];
    lines{end+1} = sprintf('Lesions (%s): MA %d, HE %d (quadrants %s), EX %d, SE %d', s1.method, ...
        s1.lesions.MA.count, s1.lesions.HE.count, mat2str(s1.hemorrhagesPerQuadrant), s1.lesions.EX.count, s1.lesions.SE.count);
    lines{end+1} = sprintf('PDR evidence: NV probability %.2f (classifier, not localised)', c.nvProbability);
    for i = 1:numel(lines), p = Paragraph(lines{i}); p.FontSize = '8.5pt'; append(d, p); end

    append(d, headed('Confidence and review'));
    if isnan(a.score), attText = a.note; else, attText = sprintf('%.2f (chance %.2f, lift %.1f) - %s', a.score, a.chanceLevel, a.lift, a.note); end
    flags = 'none'; if ~isempty(f.flagReasons), flags = strjoin(f.flagReasons, '; '); end
    lines = {f.confidenceText; ['Attention agreement: ' attText]; ['Review flags: ' flags]; ...
        sprintf('Pipeline time: %d ms (stage 0 %d, 1 %d, 2 %d, 3 %d)', result.timingMs.total, ...
        result.timingMs.stage0, result.timingMs.stage1, result.timingMs.stage2, result.timingMs.stage3)};
    for i = 1:numel(lines), p = Paragraph(lines{i}); p.FontSize = '8.5pt'; append(d, p); end

    append(d, headed('Recommendation'));
    p = Paragraph(result.recommendation); p.FontSize = '8.5pt'; append(d, p);
    foot = Paragraph(sprintf('Model %s   |   calibration fingerprint %s...   |   Screening aid, not a diagnosis. Every image is read by an eye-care professional.', ...
        result.modelVersion, result.calibrationFingerprint(1:16)));
    foot.FontSize = '7pt'; foot.Color = '#64748B'; append(d, foot);
    close(d);
    rmdir(tmp, 's');

    slim = rmfield(result, 'stage3');
    slim.stage3 = rmfield(rmfield(result.stage3, 'overlays'), 'heatReferable');
    slim.stage1 = rmfield(result.stage1, 'masks');
    jsonPath = fullfile(outDir, [result.sessionId '.json']);
    fid = fopen(jsonPath, 'w'); fwrite(fid, jsonencode(slim, 'PrettyPrint', true)); fclose(fid);
    paths = struct('pdf', pdfPath, 'json', jsonPath);
end

function h = headed(text)
    import mlreportgen.dom.*
    h = Heading3(text); h.Color = '#102A43';
end

function out = ifelse(cond, a, b)
    if cond, out = a; else, out = b; end
end
