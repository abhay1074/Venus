function fig = paretoPlot(sweep, ax)
%PARETOPLOT  Cost against missed referable cases, with the Pareto front.
%   fig = paretoPlot(sweep) for the struct from runSweep (or the decoded
%   backend/config/sweep_cache.json). Pass an axes handle as the second
%   argument to draw into the App's uiaxes.

    if nargin < 2
        fig = figure('Name', 'District sweep - Pareto front', 'Color', 'w');
        ax = axes(fig);
    else
        fig = ancestor(ax, 'figure');
    end
    runs = sweep.runs;
    if iscell(runs), runs = [runs{:}]; end
    ai = runs([runs.ai]); base = runs(~[runs.ai]);
    front = sweep.paretoFront; if iscell(front), front = [front{:}]; end
    cla(ax); hold(ax, 'on');
    scatter(ax, [ai.costInrTotal] / 1e5, [ai.missedTotal], 36, [0.06 0.46 0.43], 'filled', 'MarkerFaceAlpha', 0.55, 'DisplayName', 'AI configurations');
    scatter(ax, [base.costInrTotal] / 1e5, [base.missedTotal], 48, [0.58 0.64 0.72], '^', 'filled', 'DisplayName', 'No-AI baselines');
    plot(ax, [front.costInrTotal] / 1e5, [front.missedTotal], '-o', 'Color', [0.75 0.07 0.24], 'MarkerFaceColor', [0.75 0.07 0.24], 'LineWidth', 1.5, 'DisplayName', 'Pareto front');
    xlabel(ax, 'Programme cost, INR lakh / year'); ylabel(ax, 'Referable cases missed / year');
    title(ax, sprintf('Cameras x ophthalmologists x operating point (%d runs)', numel(runs)));
    grid(ax, 'on'); legend(ax, 'Location', 'northeast');
    if isfield(sweep, 'slideNumbers') && ~isempty(sweep.slideNumbers)
        s = sweep.slideNumbers;
        text(ax, 0.02, 0.06, sprintf('%d ophthalmologists with AI vs %d without  |  INR %.1f L vs %.1f L  |  missed %d vs %d', ...
            s.doctorsWithAi, s.doctorsWithoutAi, s.costWithAi / 1e5, s.costWithoutAi / 1e5, s.missedWithAi, s.missedWithoutAi), ...
            'Units', 'normalized', 'FontSize', 9, 'Color', [0.06 0.16 0.26]);
    end
    hold(ax, 'off');
end
