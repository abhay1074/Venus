function results = runTests()
%RUNTESTS  Run every Venus AI MATLAB test class and print a summary.
    here = fileparts(mfilename('fullpath'));
    addpath(here); addpath(fullfile(here, 'simulink')); addpath(fullfile(here, 'app'));
    suite = matlab.unittest.TestSuite.fromFolder(fullfile(here, 'tests'));
    results = run(suite);
    disp(table(results));
    fprintf('%d passed, %d failed, %d incomplete\n', nnz([results.Passed]), nnz([results.Failed]), nnz([results.Incomplete]));
end
