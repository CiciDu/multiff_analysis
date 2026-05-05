function setup_project()
    script_dir   = fileparts(mfilename('fullpath'));
    project_root = fullfile(script_dir, '..');
    addpath(genpath(project_root));
end
