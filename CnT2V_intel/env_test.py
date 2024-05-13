#!/usr/bin/python3
import sys
sys.path.append("./CnT2V_intel")
import CnT2V_intel
import warnings

##  关闭警告
warnings.filterwarnings("ignore")

##  配置参数
width = 240
height = 240
frames = 8*2
steps = 40
cn_prompt = "一只在岩石上吃竹子的熊猫"

##  处理过程回调函数
def progress_callback(index, tensor, latents):
    print("--------" + str(index+1) + "/" + str(steps) + "--------------")

##  提示输入参数
print("width=" + str(width) + " height=" + str(height) + " frames=" + str(frames) + " steps=" + str(steps))
print("Chinese input prompt: " + cn_prompt)

##  初始化翻译模型
tsl_model = CnT2V_intel.translate_model_init('./opus-mt-zh-en')

##  将中文翻译为英文
en_prompt = CnT2V_intel.translate_cn_to_en(cn_prompt, tsl_model)

##  输出英文结果
print("English output prompt: " + en_prompt)

##  载入原始模型
origin_pipe = CnT2V_intel.get_origin_model_pipe('./zeroscope_v2_576w')

##  测试基于iGPU+NPU加速器
print("Exporting IR model...")
ttv_IR_model = CnT2V_intel.export_IR_model(
                   pipe = origin_pipe, 
                   output_path = './zeroscope_v2_576w_IR', 
                   in_width = width, 
                   in_height = height,
                   in_frames = frames)
print("Starting inference with accelerator...")
CnT2V_intel.text_to_video_by_iGPU_NPU(
    prompt = en_prompt, 
    ttv_IR_model = ttv_IR_model,
    steps = steps,
    out_gif_file = "test_igpunpu.gif",
    out_mp4_file = "test_igpunpu.mp4",
    callback = progress_callback)
print("iGPU and NPU Accelerator mode inference completed!")

##  测试无加速器的
print("Import the original model...")
noacc_pipe = CnT2V_intel.noacc_model_pipeline(origin_pipe)
print("Starting inference without accelerator...")
CnT2V_intel.text_to_video_by_noacc(
    prompt = en_prompt,
    pipe = noacc_pipe,
    steps = steps,
    in_width = width, 
    in_height = height,
    in_frames = frames,
    out_gif_file = "test_noacc.gif",
    out_mp4_file = "test_noacc.mp4",
    callback = progress_callback)
print("No accelerator mode inference completed!")
