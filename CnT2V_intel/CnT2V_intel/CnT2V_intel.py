#!/usr/bin/python3
#*************************************************************
#
#   程序名称:Python模块子程序
#   程序版本:REV 0.1
#   创建日期:20240306
#   设计编写:王祥福
#   作者邮箱:rainhenry@savelife-tech.com
#
#   版本修订
#       REV 0.1   20240306      王祥福    创建文档
#
#   设计参考
#       [1]  https://docs.openvino.ai/2023.3/notebooks/253-zeroscope-text2video-with-output.html
#            https://docs.openvino.ai/2024/notebooks/253-zeroscope-text2video-with-output.html
#       [2]  https://huggingface.co/cerspense/zeroscope_v2_576w
#
#*************************************************************
##  导入模块
import os
import gc
from typing import Optional, Union, List, Callable
import diffusers
import transformers
import numpy as np
import IPython
import ipywidgets as widgets
import openvino.torch
import torch
import PIL
import gradio as gr
import openvino as ov
import openvino.properties as properties
import openvino.properties.device as device
import openvino.properties.hint as hints
import openvino.properties.streams as streams
import openvino.properties.intel_auto as intel_auto
from openvino import save_model
from pathlib import Path
import cv2
import intel_npu_acceleration_library
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import export_to_video
from nncf import compress_weights
from nncf.parameters import CompressWeightsMode

##  文字生成视频的SD管道类
class OVTextToVideoSDPipeline(diffusers.DiffusionPipeline):
    def __init__(
        self,
        vae_decoder: ov.CompiledModel,
        text_encoder: ov.CompiledModel,
        tokenizer: transformers.CLIPTokenizer,
        unet: ov.CompiledModel,
        scheduler: diffusers.schedulers.DDIMScheduler,
        vae_scale_factor,
        unet_in_channels,
        in_width: int = 0, 
        in_height: int = 0,
        in_num_frames: int = 0,
    ):
        super().__init__()

        gc.disable()   ##  关闭垃圾回收
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler
        self.vae_scale_factor = vae_scale_factor
        self.unet_in_channels = unet_in_channels
        self.width = in_width
        self.height = in_height
        self.num_frames = in_num_frames

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 9.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the video generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality videos at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate videos that are closely linked to the text `prompt`,
                usually at the expense of lower video quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (Î·) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`. Latents should be of shape
                `(batch_size, num_channel, num_frames, height, width)`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generate video. Choose between `torch.FloatTensor` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.TextToVideoSDPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            `List[np.ndarray]`: generated video frames
        """

        gc.disable()   ##  关闭垃圾回收
        num_images_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet_in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = {"generator": generator, "eta": eta}

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                gc.disable()   ##  关闭垃圾回收
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    {
                        "sample": latent_model_input,
                        "timestep": t,
                        "encoder_hidden_states": prompt_embeds,
                    }
                )[0]
                noise_pred = torch.tensor(noise_pred)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # reshape latents
                bsz, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(
                    bsz * frames, channel, width, height
                )
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(
                    bsz * frames, channel, width, height
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

                # reshape latents back
                latents = (
                    latents[None, :]
                    .reshape(bsz, frames, channel, width, height)
                    .permute(0, 2, 1, 3, 4)
                )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        video_tensor = self.decode_latents(latents)

        if output_type == "pt":
            video = video_tensor
        else:
            video = tensor2vid(video_tensor)

        if not return_dict:
            return (video,)

        return {"frames": video}

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                print(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(text_input_ids)
            prompt_embeds = prompt_embeds[0]
            prompt_embeds = torch.tensor(prompt_embeds)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids)
            negative_prompt_embeds = negative_prompt_embeds[0]
            negative_prompt_embeds = torch.tensor(negative_prompt_embeds)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        dtype,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            self.num_frames,
            self.height // self.vae_scale_factor,
            self.width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if self.height == 0 or self.width == 0 or self.num_frames == 0:
            raise ValueError(
                f"`height` or `width` or `num_frames` have not setup."
            )

        if self.height % 8 != 0 or self.width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {self.height} and {self.width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def decode_latents(self, latents):
        scale_factor = 0.18215
        latents = 1 / scale_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, height, width
        )
        image = self.vae_decoder(latents)[0]
        image = torch.tensor(image)
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        video = video.float()
        return video


##  转换模型为OpenVINO的IR格式
def convert(model: torch.nn.Module, xml_path: str, **convert_kwargs) -> Path:
    gc.disable()   ##  关闭垃圾回收
    xml_path = Path(xml_path)
    if not xml_path.exists():
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            converted_model = ov.convert_model(model, **convert_kwargs)
        ov.save_model(converted_model, xml_path)
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()
    return xml_path

##  VAE解码封装类
class VaeDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        gc.disable()   ##  关闭垃圾回收
        super().__init__()
        self.vae = vae

    def forward(self, z: torch.FloatTensor):
        return self.vae.decode(z)

##  图形格式转换
def tensor2vid(video: torch.Tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> List[np.ndarray]:
    gc.disable()   ##  关闭垃圾回收
    # This code is copied from https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/pipelines/multi_modal/text_to_video_synthesis_pipeline.py#L78
    # reshape to ncfhw
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    # unnormalize back to [0,1]
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    # prepare the final outputs
    i, c, f, h, w = video.shape
    images = video.permute(2, 3, 0, 4, 1).reshape(
        f, h, i * w, c
    )  # 1st (frames, h, batch_size, w, c) 2nd (frames, h, batch_size * w, c)
    images = images.unbind(dim=0)  # prepare a list of indvidual (consecutive frames)
    images = [(image.cpu().numpy() * 255).clip(0, 255).astype("uint8") for image in images]  # f h w c
    return images


##  导出视频为MP4文件
def export_to_mp4(video_frames: List[PIL.Image.Image], output_video_path: str = None, fps: int = 8) -> str:
    gc.disable()   ##  关闭垃圾回收
    if output_video_path is None:
        return None

    video_frames = [np.array(frame) for frame in video_frames]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = video_frames[i]
        img = np.round(img)
        img = img.astype(np.uint8)
        np.clip(img, 0, 255, out=img)
        img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        video_writer.write(img)
    return output_video_path

##  获取原始模型的管道
def get_origin_model_pipe(model_id):
    gc.disable()   ##  关闭垃圾回收
    pipe = diffusers.DiffusionPipeline.from_pretrained(model_id)
    return pipe

##  将文字生成视频的模型导出为IR模型
def export_IR_model(pipe, output_path, in_width, in_height, in_frames):
    gc.disable()   ##  关闭垃圾回收
    unet = pipe.unet
    unet.eval()
    vae = pipe.vae
    vae.eval()
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    tokenizer = pipe.tokenizer
    scheduler = pipe.scheduler
    vae_scale_factor = pipe.vae_scale_factor
    unet_in_channels = pipe.unet.config.in_channels
    sample_width = in_width // vae_scale_factor
    sample_height = in_height // vae_scale_factor

    ##  导出unet
    unet_xml_path = convert(
        unet,
        output_path + "/unet.xml",
        example_input={
            "sample": torch.randn(2, 4, 2, int(sample_height // 2), int(sample_width // 2)),
            "timestep": torch.tensor(1),
            "encoder_hidden_states": torch.randn(2, 77, 1024),
        },
        input=[
            ("sample", (2, 4, in_frames, sample_height, sample_width)),
            ("timestep", ()),
            ("encoder_hidden_states", (2, 77, 1024)),
        ],
    )
    
    ##  导出vae
    vae_decoder_xml_path = convert(
        VaeDecoderWrapper(vae),
        output_path + "/vae.xml",
        example_input=torch.randn(2, 4, 32, 32),
        input=((in_frames, 4, sample_height, sample_width)),
    )

    ##  压缩
    core = ov.Core()
    unet_model = core.read_model(unet_xml_path)
    ##unet_model = compress_weights(unet_model,  mode=CompressWeightsMode.INT4_ASYM, group_size=64, ratio=1.0)  ##  实测效果一般，取消UNET的量化压缩后OK
    vae_model = core.read_model(vae_decoder_xml_path)
    vae_model  = compress_weights(vae_model,  mode=CompressWeightsMode.INT8_ASYM)

    ##  编译成iGPU+NPU上
    ov_unet = core.compile_model(unet_model, device_name='MULTI:GPU,NPU', config={hints.performance_mode: hints.PerformanceMode.THROUGHPUT})
    ov_vae_decoder = core.compile_model(vae_model, device_name='MULTI:GPU,NPU', config={hints.performance_mode: hints.PerformanceMode.THROUGHPUT})
    ov_text_encoder = intel_npu_acceleration_library.compile(text_encoder)

    ##  输出结果
    ret = [ov_unet, ov_vae_decoder, ov_text_encoder, tokenizer, scheduler, vae_scale_factor, unet_in_channels, in_width, in_height, in_frames]
    return ret

##  获取无加速器模型的管道
def noacc_model_pipeline(pipe):
    ##pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to('cpu')
    return pipe

##  传统推理,无加速
def text_to_video_by_noacc(prompt, pipe, steps, in_width, in_height, in_frames, out_gif_file, out_mp4_file, callback=None):
    video_frames = pipe(prompt, num_inference_steps=steps, height=in_height, width=in_width, num_frames=in_frames, callback=callback).frames

    ##  导出视频文件
    export_to_video(video_frames=video_frames[0], output_video_path=out_mp4_file)

    ##  导出动图文件
    video_frames_float32 = np.clip(video_frames[0]*255.0, 0, 255)
    video_frames_uint8 = np.uint8(video_frames_float32)
    images = [PIL.Image.fromarray(frame) for frame in video_frames_uint8]
    images[0].save(out_gif_file, save_all=True, append_images=images[1:], duration=125, loop=0)
    
##  通过iGPU和NPU混合执行推理
def text_to_video_by_iGPU_NPU(prompt, ttv_IR_model, steps, out_gif_file, out_mp4_file, callback=None):
    gc.disable()   ##  关闭垃圾回收

    ##  获取参数
    ov_unet = ttv_IR_model[0]
    ov_vae_decoder = ttv_IR_model[1]
    ov_text_encoder = ttv_IR_model[2]
    tokenizer = ttv_IR_model[3]
    scheduler = ttv_IR_model[4]
    vae_scale_factor = ttv_IR_model[5]
    unet_in_channels = ttv_IR_model[6]
    in_width = ttv_IR_model[7]
    in_height =ttv_IR_model[8]
    in_frames = ttv_IR_model[9]

    ov_pipe = OVTextToVideoSDPipeline(
                  vae_decoder = ov_vae_decoder, 
                  text_encoder = ov_text_encoder, 
                  tokenizer = tokenizer, 
                  unet = ov_unet, 
                  scheduler = scheduler,
                  vae_scale_factor = vae_scale_factor,
                  unet_in_channels = unet_in_channels,
                  in_width=in_width, 
                  in_height=in_height, 
                  in_num_frames=in_frames)
    frames = ov_pipe(prompt, num_inference_steps=steps, callback=callback)['frames']

    ##  保存为gif动图文件
    images = [PIL.Image.fromarray(frame) for frame in frames]
    images[0].save(out_gif_file, save_all=True, append_images=images[1:], duration=125, loop=0)

    ##  保存为视频文件
    video_path = export_to_mp4(video_frames=frames, output_video_path=out_mp4_file)     

##  载入翻译模型
def translate_model_init(model_id):
    gc.disable()   ##  关闭垃圾回收
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_id)
    ret = [tokenizer, model]
    return ret

##  执行翻译
def translate_cn_to_en(in_text, tsl_model):
    gc.disable()   ##  关闭垃圾回收
    tokenizer = tsl_model[0]
    model = tsl_model[1]
    inputs = tokenizer(in_text, return_tensors='pt')
    pred = model.generate(**inputs)
    output = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
    return output
