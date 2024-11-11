csv_file_path = '/Users/dusiyi/Documents/Multifirefly-Project/multiff_analysis/methods/neural_data_analysis/MATLAB_processing/data_table.csv'

% Read the CSV file into a table
opts = detectImportOptions(csv_file_path, 'Delimiter', ',');

% Set the first column as row names
opts.RowNamesColumn = 0;

% Read the table with the specified options
data_table = readtable(csv_file_path, opts);

% Add a new index column
data_table.Index = (1:height(data_table))';

% Move the new index column to the first position
data_table = movevars(data_table, 'Index', 'Before', data_table.Properties.VariableNames{1});

% Display the first few rows of the modified table
disp(head(data_table));

% iterate through each row of data_table
for i = 1:height(data_table)
    % get the local_file_path and hdrive_file_path
    local_file_path = data_table{i, 'local_path'}{1}
    hdrive_file_path = data_table{i, 'hdrive_path'}{1}
    
    % get the list of files in the local_file_path
    files = dir(local_file_path);

    % iterate through each file 
    for j = 1:length(files)
        file_name = files(j).name
        
        % if '_ead' is in file_name, skip the file
        if contains(file_name, '_ead')
            continue
        end

        if contains(file_name, '.plx')
            fname = fullfile(local_file_path, file_name);
            disp(fname)

            [~, ts, sv, freq] = plx_event_ts_modified(fname, 257); 
            ts_s = ts/freq; ts_s = ts_s.'; sv = sv.'; ts = ts.';
            for_aligning_data = table(sv, ts, ts_s);
            writetable(for_aligning_data, fullfile(local_file_path, 'neural_data_alignment.txt'))
            type(fullfile(local_file_path,'neural_data_alignment.txt'))

        end
    end
end
