# #foggy

    #wsco+MT
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/foggy.yaml --model-dir source_model/cityscape_baseline/model_final.pth --method wsco --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+IRG
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/foggy.yaml --model-dir source_model/cityscape_baseline/model_final.pth --method wsco+IRG --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+LODS
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/style/foggy.yaml --model-dir source_model/cityscape_baseline/model_final.pth --method wsco+LODS --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+LPLD
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/foggy.yaml --model-dir source_model/cityscape_baseline/model_final.pth --method wsco+LPLD --merge merge --cmaweight 0.5 --temp 0.1

# #sim

    #wsco+MT
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/sim.yaml --model-dir source_model/sim_baseline/model_final.pth --method wsco --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+IRG
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/sim.yaml --model-dir source_model/sim_baseline/model_final.pth --method wsco+IRG --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+LODS
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/style/sim.yaml --model-dir source_model/sim_baseline/model_final.pth --method wsco+LODS --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+LPLD
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/sim.yaml --model-dir source_model/sim_baseline/model_final.pth --method wsco+LPLD --merge merge --cmaweight 0.5 --temp 0.1


# #kitti

    #wsco+MT
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/kitti.yaml --model-dir source_model/kitti_baseline/model_final.pth --method wsco --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+IRG
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/kitti.yaml --model-dir source_model/kitti_baseline/model_final.pth --method wsco+IRG --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+LODS
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/style/kitti.yaml --model-dir source_model/kitti_baseline/model_final.pth --method wsco+LODS --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+LPLD
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/kitti.yaml --model-dir source_model/kitti_baseline/model_final.pth --method wsco+LPLD --merge merge --cmaweight 0.5 --temp 0.1

#water

    #wsco+MT (res50)
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/water.yaml --model-dir source_model/water/model_final_50.pth --method wsco --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+MT (res101)
    python tools/train_st_sfda_net.py --config-file configs/wsco/res101/water.yaml --model-dir source_model/water/model_final_101.pth --method wsco --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+IRG
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/water.yaml --model-dir source_model/water/model_final_50.pth --method wsco+IRG --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+LODS
    python tools/train_st_sfda_net.py --config-file configs/wsco/res101/water.yaml --model-dir source_model/water/model_final_101.pth --method wsco+LODS --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+LPLD
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/water.yaml --model-dir source_model/water/model_final_50.pth --method wsco+LPLD --merge merge --cmaweight 0.5 --temp 0.1

# #clipart

    #wsco+MT (res50)
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/clipart.yaml --model-dir source_model/clipart/model_final_50.pth --method wsco --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+MT (res101)
    python tools/train_st_sfda_net.py --config-file configs/wsco/res101/clipart.yaml --model-dir source_model/clipart/model_final_101.pth --method wsco --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+IRG
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/clipart.yaml --model-dir source_model/clipart/model_final_50.pth --method wsco+IRG --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+LODS
    python tools/train_st_sfda_net.py --config-file configs/wsco/res101/clipart.yaml --model-dir source_model/VOC_baseline/model_final_101.pth --method wsco+LODS --merge merge --cmaweight 0.5 --temp 0.1
    #wsco+LPLD
    python tools/train_st_sfda_net.py --config-file configs/wsco/res50/clipart.yaml --model-dir source_model/clipart/model_final_50.pth --method wsco+LPLD --merge merge --cmaweight 0.5 --temp 0.1

