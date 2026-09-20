classdef DRScreenApp < handle
    %DRSCREENAPP  Venus AI desktop app (uifigure): Capture, Result, Review, District.
    %   app = DRScreenApp()          loads models via drscreen.loadModels
    %   app = DRScreenApp(models)    reuse loaded models
    %
    %   Four tabs, as the architecture's App Designer app:
    %     Capture / Upload  choose a fundus image, see the Stage 0 verdict and reason
    %     Result            grade, referable, calibrated P, overlays with toggles,
    %                       rule trace, attention agreement, tier, PDF
    %     Review queue      flagged cases of this session sorted by P(referable)
    %     District          edit params, run one year AI vs no-AI, run the sweep,
    %                       Pareto plot with the four slide numbers
    %   Every button calls the same +drscreen functions the tests call.

    properties
        Models
        Fig
        Tabs
        ImagePath
        Result
        Queue = struct('sessionId', {}, 'pReferable', {}, 'cnnGrade', {}, 'ruleGrade', {}, 'flag', {}, 'tier', {}, 'reasons', {})
        UI = struct()
    end

    methods
        function app = DRScreenApp(models)
            if nargin < 1, models = drscreen.loadModels(); end
            app.Models = models;
            app.Fig = uifigure('Name', 'Venus AI - Diabetic Retinopathy Screening', 'Position', [80 80 1240 800], 'Color', [0.97 0.98 0.99]);
            app.Tabs = uitabgroup(app.Fig, 'Position', [10 10 1220 780]);
            app.buildCapture(); app.buildResult(); app.buildReview(); app.buildDistrict();
        end

        % ------------------------------------------------------- Capture --
        function buildCapture(app)
            tab = uitab(app.Tabs, 'Title', 'Capture / Upload');
            uilabel(tab, 'Text', 'Portable fundus camera or phone upload -> Stage 0 gate', 'Position', [20 730 600 22], 'FontSize', 14, 'FontWeight', 'bold');
            uibutton(tab, 'Text', 'Select image...', 'Position', [20 690 140 30], 'ButtonPushedFcn', @(~, ~) app.pickImage());
            uibutton(tab, 'Text', 'Screen this image', 'Position', [170 690 160 30], 'BackgroundColor', [0.06 0.46 0.43], 'FontColor', 'w', 'ButtonPushedFcn', @(~, ~) app.screen());
            app.UI.captureImage = uiimage(tab, 'Position', [20 200 480 480], 'ScaleMethod', 'fit');
            app.UI.captureVerdict = uitextarea(tab, 'Position', [520 400 680 280], 'Editable', 'off', 'FontSize', 13, ...
                'Value', {'Stage 0 decides three things before any model runs: is this a retina, is it gradable, does it need enhancement.'});
            app.UI.intakeSymptoms = uilistbox(tab, 'Position', [520 250 300 120], 'Multiselect', 'on', ...
                'Items', {'blurred_vision', 'floaters', 'sudden_vision_loss'}, 'Value', {});
            uilabel(tab, 'Text', 'Symptoms (optional)', 'Position', [520 372 200 22]);
            uilabel(tab, 'Text', 'HbA1c', 'Position', [840 350 100 22]); app.UI.hba1c = uieditfield(tab, 'numeric', 'Position', [840 325 100 24], 'Value', 0);
            app.UI.pregnant = uicheckbox(tab, 'Text', 'Pregnant', 'Position', [840 290 120 22]);
            app.UI.tta = uicheckbox(tab, 'Text', 'Test-time augmentation (5x)', 'Position', [840 260 220 22]);
        end

        function pickImage(app)
            [f, p] = uigetfile({'*.jpg;*.jpeg;*.png;*.webp', 'Fundus photographs'}, 'Select a fundus photograph', ...
                fullfile(drscreen.repoRoot(), 'samples'));
            if isequal(f, 0), return; end
            app.ImagePath = fullfile(p, f);
            app.UI.captureImage.ImageSource = app.ImagePath;
            app.UI.captureVerdict.Value = {sprintf('%s selected. Press "Screen this image".', f)};
        end

        function screen(app)
            if isempty(app.ImagePath), uialert(app.Fig, 'Select an image first.', 'No image'); return; end
            intake = struct('symptoms', {app.UI.intakeSymptoms.Value}, 'hba1c', app.UI.hba1c.Value, 'pregnant', app.UI.pregnant.Value);
            d = uiprogressdlg(app.Fig, 'Title', 'Screening', 'Message', 'Stage 0 -> 1 -> 2 -> 3', 'Indeterminate', 'on');
            try
                app.Result = drscreen.screenImage(app.ImagePath, app.Models, struct('intake', intake, 'tta', app.UI.tta.Value, 'report', true));
            catch err
                close(d); uialert(app.Fig, err.message, 'Screening failed'); return;
            end
            close(d);
            r = app.Result;
            if ~r.accepted
                app.UI.captureVerdict.Value = {['RETAKE / REFUSED: ' r.stopReason], sprintf('Tier %s - %s (%s)', r.stage5.tier, r.stage5.label, r.stage5.deadlineText), ...
                    'Nothing diagnostic ran.'};
                return;
            end
            q = r.stage0.quality;
            app.UI.captureVerdict.Value = {sprintf('Gate: fundus P = %.3f. Quality: %s (score %.2f)%s', r.stage0.modality.fundusProbability, ...
                q.label, q.score, ifelse(q.enhanced, ' - enhanced', '')), sprintf('Result: %s. Open the Result tab.', ...
                ifelse(r.stage2.fusion.referable, 'REFERABLE', 'not referable'))};
            app.showResult(); app.pushQueue();
            app.Tabs.SelectedTab = app.Tabs.Children(2);
        end

        % -------------------------------------------------------- Result --
        function buildResult(app)
            tab = uitab(app.Tabs, 'Title', 'Result');
            app.UI.verdict = uilabel(tab, 'Text', 'Awaiting an image', 'Position', [20 720 900 40], 'FontSize', 24, 'FontWeight', 'bold');
            app.UI.verdictSub = uilabel(tab, 'Text', '', 'Position', [20 690 1100 24], 'FontSize', 13);
            app.UI.overlaySelect = uidropdown(tab, 'Position', [20 650 260 26], 'Items', ...
                {'Lesion overlay', 'Grad-CAM referable', 'Grad-CAM grade', 'Vessels', 'Stage 0 output', 'Original'}, ...
                'ValueChangedFcn', @(~, ~) app.showOverlay());
            app.UI.resultImage = uiimage(tab, 'Position', [20 100 540 540], 'ScaleMethod', 'fit');
            app.UI.evidence = uitextarea(tab, 'Position', [580 100 620 540], 'Editable', 'off', 'FontName', 'Consolas', 'FontSize', 12);
            uibutton(tab, 'Text', 'Open PDF report', 'Position', [580 650 160 30], 'ButtonPushedFcn', @(~, ~) app.openReport());
        end

        function showResult(app)
            r = app.Result; f = r.stage2.fusion;
            txt = ifelse(f.referable, 'Referable DR', 'Not referable');
            if f.flagForReview, txt = [txt '  -  human review']; end
            app.UI.verdict.Text = txt;
            app.UI.verdict.FontColor = ifelse(f.flagForReview, [0.71 0.33 0.04], ifelse(f.referable, [0.75 0.07 0.24], [0.06 0.46 0.43]));
            app.UI.verdictSub.Text = sprintf('ICDR grade %d - %s | %s | tier %s - %s, %s', f.grade, f.gradeLabel, f.confidenceText, ...
                r.stage5.tier, r.stage5.label, r.stage5.deadlineText);
            a = r.stage3.attentionAgreement;
            lines = [{'RULE GRADER - ICDR CRITERIA'}; cellfun(@(x) ['  - ' x], r.stage2.rule.trace(:), 'UniformOutput', false); {''; ...
                sprintf('Five-grade probabilities: %s', mat2str(round(r.stage2.cnn.gradeProbabilities, 2))); ...
                sprintf('Lesions (%s): MA %d  HE %d (quadrants %s)  EX %d  SE %d', r.stage1.method, r.stage1.lesions.MA.count, ...
                    r.stage1.lesions.HE.count, mat2str(r.stage1.hemorrhagesPerQuadrant), r.stage1.lesions.EX.count, r.stage1.lesions.SE.count); ...
                sprintf('P(NV) %.2f (classifier, not localised)', r.stage2.cnn.nvProbability); ''; ...
                sprintf('Attention agreement: %s', ifelse(isnan(a.score), a.note, sprintf('%.2f (chance %.2f, lift %.1f) - %s', a.score, a.chanceLevel, a.lift, a.note))); ...
                ['Review flags: ' ifelse(isempty(f.flagReasons), 'none', strjoin(f.flagReasons, '; '))]; ''; ...
                sprintf('Timing: %d ms total (S0 %d, S1 %d, S2 %d, S3 %d)', r.timingMs.total, r.timingMs.stage0, r.timingMs.stage1, r.timingMs.stage2, r.timingMs.stage3); ''; ...
                r.recommendation}];
            app.UI.evidence.Value = lines;
            app.showOverlay();
        end

        function showOverlay(app)
            if isempty(app.Result) || ~app.Result.accepted, return; end
            o = app.Result.stage3.overlays;
            switch app.UI.overlaySelect.Value
                case 'Lesion overlay', img = o.lesions;
                case 'Grad-CAM referable', img = o.gradcamReferable;
                case 'Grad-CAM grade', img = o.gradcamGrade;
                case 'Vessels', img = o.vessels;
                case 'Stage 0 output', img = o.enhanced;
                otherwise, img = o.original;
            end
            app.UI.resultImage.ImageSource = img;
        end

        function openReport(app)
            if isempty(app.Result) || ~isfield(app.Result, 'report'), return; end
            open(app.Result.report.pdf);
        end

        % -------------------------------------------------------- Review --
        function buildReview(app)
            tab = uitab(app.Tabs, 'Title', 'Review queue');
            uilabel(tab, 'Text', 'Flagged cases first, then by P(referable). Both grades shown.', 'Position', [20 730 800 22], 'FontSize', 14, 'FontWeight', 'bold');
            app.UI.queueTable = uitable(tab, 'Position', [20 60 1180 650], 'ColumnName', {'Session', 'P(referable)', 'CNN grade', 'Rule grade', 'Flag', 'Tier', 'Why'});
        end

        function pushQueue(app)
            r = app.Result;
            app.Queue(end+1) = struct('sessionId', r.sessionId, 'pReferable', r.stage2.fusion.pReferable, 'cnnGrade', r.stage2.cnn.grade, ...
                'ruleGrade', r.stage2.rule.grade, 'flag', r.stage2.fusion.flagForReview, 'tier', r.stage5.tier, ...
                'reasons', strjoin(r.stage2.fusion.flagReasons, '; '));
            q = app.Queue;
            [~, order] = sortrows([-double([q.flag])', -[q.pReferable]']);
            q = q(order);
            app.UI.queueTable.Data = [{q.sessionId}', num2cell([q.pReferable]'), num2cell([q.cnnGrade]'), num2cell([q.ruleGrade]'), ...
                num2cell([q.flag]'), {q.tier}', {q.reasons}'];
        end

        % ------------------------------------------------------ District --
        function buildDistrict(app)
            tab = uitab(app.Tabs, 'Title', 'District');
            addpath(fullfile(drscreen.repoRoot(), 'matlab', 'simulink'));
            uilabel(tab, 'Text', 'District screening programme - one simulated year, coupled to the measured operating point', ...
                'Position', [20 730 900 22], 'FontSize', 14, 'FontWeight', 'bold');
            names = {'annualPatients', 'phcs', 'camerasPerPhc', 'ophthalmologists', 'teleReviewMinutes', 'inPersonMinutes', 'doctorHoursPerDay', 'humanReaderSensitivity'};
            p = districtParams();
            app.UI.paramFields = struct();
            for i = 1:numel(names)
                uilabel(tab, 'Text', names{i}, 'Position', [20 + 150 * (i - 1), 700, 140, 20]);
                app.UI.paramFields.(names{i}) = uieditfield(tab, 'numeric', 'Position', [20 + 150 * (i - 1), 675, 130, 24], 'Value', p.(names{i}));
            end
            uibutton(tab, 'Text', 'Run one year, AI vs no-AI', 'Position', [20 630 220 30], 'ButtonPushedFcn', @(~, ~) app.runYear());
            uibutton(tab, 'Text', 'Run sweep (Pareto)', 'Position', [250 630 180 30], 'ButtonPushedFcn', @(~, ~) app.runSweepUi());
            uibutton(tab, 'Text', 'Open Simulink model', 'Position', [440 630 180 30], 'ButtonPushedFcn', @(~, ~) app.openSimulink());
            app.UI.districtText = uitextarea(tab, 'Position', [20 60 520 560], 'Editable', 'off', 'FontName', 'Consolas', 'FontSize', 12);
            app.UI.districtAxes = uiaxes(tab, 'Position', [560 60 640 560]);
        end

        function p = districtParamsFromUi(app)
            p = districtParams();
            names = fieldnames(app.UI.paramFields);
            for i = 1:numel(names), p.(names{i}) = app.UI.paramFields.(names{i}).Value; end
            p.operatorsPerPhc = p.camerasPerPhc;
        end

        function runYear(app)
            p = app.districtParamsFromUi();
            ai = simulateDistrict(p); p.aiEnabled = false; base = simulateDistrict(p);
            fmt = @(r) {sprintf('  wait mean %.1f d, p95 %.1f d', r.waitCaptureToResultDays.mean, r.waitCaptureToResultDays.p95), ...
                sprintf('  utilisation %.0f%%, backlog %d', 100 * r.ophthalmologistUtilisation, r.backlogAtEnd), ...
                sprintf('  sent to doctor %d (%.0f%%)', r.sentToOphthalmologist, 100 * r.sentFraction), ...
                sprintf('  missed by AI %d / reader %d / backlog %d  -> programme sensitivity %.0f%%', r.referableMissedByAi, ...
                    r.referableMissedByReader, r.referableUnresultedAtYearEnd, 100 * r.programmeSensitivity), ...
                sprintf('  doctor hours %.0f, cost INR %.1f L (INR %.0f / patient)', r.doctorHours, r.costInrTotal / 1e5, r.costInrPerScreenedPatient)};
            app.UI.districtText.Value = [{'WITH AI TRIAGE'}; fmt(ai)'; {''; 'BASELINE: every image read by a human'}; fmt(base)'; {''; ...
                sprintf('Doctor hours saved %.0f, cost saved INR %.1f L, p95 wait change %.1f d', base.doctorHours - ai.doctorHours, ...
                    (base.costInrTotal - ai.costInrTotal) / 1e5, ai.waitCaptureToResultDays.p95 - base.waitCaptureToResultDays.p95)}];
        end

        function runSweepUi(app)
            d = uiprogressdlg(app.Fig, 'Title', 'Sweep', 'Message', 'cameras x doctors x operating point, full year each', 'Indeterminate', 'on');
            sweep = runSweep(struct('useParallel', license('test', 'Distrib_Computing_Toolbox')));
            close(d);
            paretoPlot(sweep, app.UI.districtAxes);
            if ~isempty(sweep.slideNumbers)
                s = sweep.slideNumbers;
                app.UI.districtText.Value = [app.UI.districtText.Value; {''; 'SWEEP - the four slide numbers'; ...
                    sprintf('  ophthalmologists needed: %d with AI vs %d without', s.doctorsWithAi, s.doctorsWithoutAi); ...
                    sprintf('  programme cost / year: INR %.1f L vs %.1f L', s.costWithAi / 1e5, s.costWithoutAi / 1e5); ...
                    sprintf('  referable cases missed: %d vs %d (limit %d)', s.missedWithAi, s.missedWithoutAi, sweep.constraints.maxMissed); ...
                    sprintf('  AI operating point: %.1f%% sensitivity', 100 * s.operatingPointWithAi)}];
            end
        end

        function openSimulink(app) %#ok<MANU>
            model = fullfile(drscreen.repoRoot(), 'matlab', 'simulink', 'district_screening.slx');
            if ~isfile(model), buildDistrictModel(districtParams()); end
            open_system(model);
        end
    end
end

function out = ifelse(c, a, b)
    if c, out = a; else, out = b; end
end
