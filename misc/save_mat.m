function save_mat(obj_idx, path, all_boxes)

    save([path '/detections_' sprintf('%02d',obj_idx) '.mat'],'all_boxes')
end