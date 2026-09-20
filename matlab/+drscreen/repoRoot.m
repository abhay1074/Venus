function root = repoRoot()
%REPOROOT  The repository root (the folder containing matlab/, backend/, frontend/).
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
