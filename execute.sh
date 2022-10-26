CUDA_VISIBLE_DEVICES=0,1,2,4,5 python run_sample.py \
    --voc12_root public/val_image/ \
    --num_workers 2 \
    --train_list  data/public/val.json\
    --val_list data/public/val.json \
    --infer_list data/public/val.json \
    --cam_out_dir result/try_1/validation/cam \
    --ir_label_out_dir result/try_1/validation/ir_label \
    --sem_seg_out_dir result/try_1/validation/sem_seg \
    --ins_seg_out_dir result/try_1/validation/ins_seg 2>&1 | tee try_1_validation.log
