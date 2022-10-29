CUDA_VISIBLE_DEVICES=0,1,2,4,5 python run_sample.py \
    --voc12_root public/test_image/ \
    --num_workers 8 \
    --train_list  data/public/test.json\
    --val_list data/public/test.json \
    --infer_list data/public/tset.json \
    --cam_out_dir result/try_1/test/cam \
    --ir_label_out_dir result/try_1/test/ir_label \
    --sem_seg_out_dir result/try_1/test/sem_seg \
    --ins_seg_out_dir result/try_1/test/ins_seg 2>&1 | tee try_1_test.log
