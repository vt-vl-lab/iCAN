function all_boxes = load_mat(obj_idx, path)
    
    load([path '/detections_' sprintf('%02d',obj_idx) '.mat'])
end