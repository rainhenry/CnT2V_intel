/*************************************************************

    程序名称:基于Python3原生C接口的AI C++类(阻塞)
    程序版本:REV 0.1
    创建日期:20240306
    设计编写:王祥福
    作者邮箱:rainhenry@savelife-tech.com

    版本修订
        REV 0.1   20240306      王祥福    创建文档

*************************************************************/

//  包含头文件
#include <stdio.h>
#include <stdlib.h>

#include "CPyAI.h"

#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_3kcompat.h>

//  初始化全局变量
bool CPyAI::Py_Initialize_flag = false;

//  构造函数
CPyAI::CPyAI()
{
    //  当没有初始化过
    if(!Py_Initialize_flag)
    {
        if(!Py_IsInitialized())
        {
            Py_Initialize();
            import_array_init();
        }
        Py_Initialize_flag = true;   //  标记已经初始化
    }

    //  配置参数
    cur_infer_args.args_valid   = false;
    cur_infer_args.width        = 0;
    cur_infer_args.height       = 0;
    cur_infer_args.total_frames = 0;

    //  控制权状态
    py_gil_st                   = -1;   

    //  python相关的私有数据
    py_cnt2v_module             = nullptr;

    //  翻译相关私有数据
    py_tsl_model_init           = nullptr;
    py_tsl_model_handle         = nullptr;       
    py_tsl_ex                   = nullptr;                

    //  文字生成视频相关私有数据
    py_ttv_model_pipe           = nullptr;
    py_ttv_model_pipe_handle    = nullptr;
    py_ttv_igpunpy_conv_ir      = nullptr;  
    py_ttv_igpunpu_handle       = nullptr;   
    py_ttv_igpunpu_ex           = nullptr;       
    py_ttv_noacc_pipe_init      = nullptr;  
    py_ttv_noacc_pipe_handle    = nullptr;
    py_ttv_noacc_ex             = nullptr;         
}

//  析构函数
CPyAI::~CPyAI()
{
    //  此处不可以调用Release，因为Python环境实际运行所在线程
    //  不一定和构造该类对象是同一个线程
}

//  释放资源
//  注意！该释放必须和执行本体在同一线程中！
void CPyAI::Release(void)
{

    if(py_ttv_noacc_ex != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_ttv_noacc_ex));
    }
    if(py_ttv_noacc_pipe_handle != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_ttv_noacc_pipe_handle));
    }
    if(py_ttv_noacc_pipe_init != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_ttv_noacc_pipe_init));
    }
    if(py_ttv_igpunpu_ex != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_ttv_igpunpu_ex));
    }
    if(py_ttv_igpunpu_handle != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_ttv_igpunpu_handle));
    }
    if(py_ttv_igpunpy_conv_ir != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_ttv_igpunpy_conv_ir));
    }
    if(py_ttv_model_pipe_handle != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_ttv_model_pipe_handle));
    }
    if(py_ttv_model_pipe != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_ttv_model_pipe));
    }


    if(py_tsl_ex != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_tsl_ex));
    }
    if(py_tsl_model_handle != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_tsl_model_handle));
    }
    if(py_tsl_model_init != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_tsl_model_init));
    }

    if(py_cnt2v_module != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_cnt2v_module));
    }

    if(py_gil_st != -1)
    {
        PyGILState_Release(static_cast<PyGILState_STATE>(py_gil_st));
    }

    if(Py_Initialize_flag)
    {
        //  程序退出时，由操作系统自动释放
        //Py_Finalize();
        //Py_Initialize_flag = false;   //  标记未初始化
    }
}

//  为了兼容Python C的原生API，独立封装numpy的C初始化接口
int CPyAI::import_array_init(void)
{
    import_array()
    return 0;
}

//  初始化
//  注意！该初始化必须和执行本体在同一线程中！
void CPyAI::Init(void)
{
    //  开启Python线程支持
    PyEval_InitThreads();
    PyEval_SaveThread();

    //  检测当前线程是否拥有GIL
    int ck = PyGILState_Check() ;
    if (!ck)
    {
        PyGILState_STATE state = PyGILState_Ensure(); //  如果没有GIL，则申请获取GIL
        py_gil_st = state;       //  定义于 /usr/include/python3.10/pystate.h 文件 94行，为枚举类型，可用int类型转存
    }

    //  构造基本Python环境
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../CnT2V_intel')");

    //  载入CnT2V_intel.py文件
    py_cnt2v_module = static_cast<void*>(PyImport_ImportModule("CnT2V_intel"));

    //  检查是否成功
    if(py_cnt2v_module == nullptr)
    {
        printf("[Error] py_cnt2v_module == null!!\n");
        return;
    }

    //  初始化翻译模型
    py_tsl_model_init   = static_cast<void*>(PyObject_GetAttrString(static_cast<PyObject*>(py_cnt2v_module), "translate_model_init"));
    Py_IncRef(static_cast<PyObject*>(py_tsl_model_init));
    PyObject* py_tsl_args = PyTuple_New(1);
    PyTuple_SetItem(py_tsl_args, 0, Py_BuildValue("s", CPYAI_TSL_MODEL_PATH));
    py_tsl_model_handle = static_cast<void*>(PyObject_CallObject(static_cast<PyObject*>(py_tsl_model_init), py_tsl_args));

    if(py_tsl_model_handle == nullptr)
    {
        printf("[Error] py_tsl_model_handle == null!!\n");
        return;
    }
    Py_IncRef(static_cast<PyObject*>(py_tsl_model_handle));

    //  载入执行翻译本体
    py_tsl_ex = static_cast<void*>(PyObject_GetAttrString(static_cast<PyObject*>(py_cnt2v_module), "translate_cn_to_en"));
    Py_IncRef(static_cast<PyObject*>(py_tsl_ex));

    //  获取原始模型管道
    py_ttv_model_pipe   = static_cast<void*>(PyObject_GetAttrString(static_cast<PyObject*>(py_cnt2v_module), "get_origin_model_pipe"));
    Py_IncRef(static_cast<PyObject*>(py_ttv_model_pipe));
    PyObject* py_ttv_args = PyTuple_New(1);
    PyTuple_SetItem(py_ttv_args, 0, Py_BuildValue("s", CPYAI_TTV_MODEL_PATH));
    py_ttv_model_pipe_handle = static_cast<void*>(PyObject_CallObject(static_cast<PyObject*>(py_ttv_model_pipe), py_ttv_args));

    if(py_ttv_model_pipe_handle == nullptr)
    {
        printf("[Error] py_ttv_model_pipe_handle == null!!\n");
        return;
    }
    Py_IncRef(static_cast<PyObject*>(py_ttv_model_pipe_handle));
    
    //  载入模型转换函数
    py_ttv_igpunpy_conv_ir = static_cast<void*>(PyObject_GetAttrString(static_cast<PyObject*>(py_cnt2v_module), "export_IR_model"));
    Py_IncRef(static_cast<PyObject*>(py_ttv_igpunpy_conv_ir));

    //  载入基于iGPU+NPU加速器的推理函数
    py_ttv_igpunpu_ex = static_cast<void*>(PyObject_GetAttrString(static_cast<PyObject*>(py_cnt2v_module), "text_to_video_by_iGPU_NPU"));
    Py_IncRef(static_cast<PyObject*>(py_ttv_igpunpu_ex));

    //  初始化无加速器的管道
    py_ttv_noacc_pipe_init   = static_cast<void*>(PyObject_GetAttrString(static_cast<PyObject*>(py_cnt2v_module), "noacc_model_pipeline"));
    Py_IncRef(static_cast<PyObject*>(py_ttv_noacc_pipe_init));
    py_ttv_args = PyTuple_New(1);
    PyTuple_SetItem(py_ttv_args, 0, static_cast<PyObject*>(py_ttv_model_pipe_handle));
    py_ttv_noacc_pipe_handle = static_cast<void*>(PyObject_CallObject(static_cast<PyObject*>(py_ttv_noacc_pipe_init), py_ttv_args));

    if(py_ttv_noacc_pipe_handle == nullptr)
    {
        printf("[Error] py_ttv_noacc_pipe_handle == null!!\n");
        return;
    }
    Py_IncRef(static_cast<PyObject*>(py_ttv_noacc_pipe_handle));

    //  载入无加速器的推理函数
    py_ttv_noacc_ex = static_cast<void*>(PyObject_GetAttrString(static_cast<PyObject*>(py_cnt2v_module), "text_to_video_by_noacc"));
    Py_IncRef(static_cast<PyObject*>(py_ttv_noacc_ex));

    //  查询已有导出IR模型参数
    SInferenceArgs args;
    int re = GetLocalIRargs(&args);

    //  当已有IR模型参数查询成功
    if(re == 0)
    {
        //  保存参数
        this->cur_infer_args = args;
    }
}

//  执行中文到英文的翻译
std::string CPyAI::Translate_Cn2En_Ex(const char* prompt)
{
    //  返回字符串
    std::string re_str;

    //  构造输入数据
    PyObject* py_args = PyTuple_New(2);

    //  第一个参数为关键字
    PyObject* py_prompt = Py_BuildValue("s", prompt);
    PyTuple_SetItem(py_args, 0, py_prompt);

    //  第二个参数为模型句柄
    PyTuple_SetItem(py_args, 1, static_cast<PyObject*>(py_tsl_model_handle));

    //  执行
    PyObject* py_ret = PyObject_CallObject(static_cast<PyObject*>(py_tsl_ex), py_args);

    //  检查
    if(py_ret == nullptr)
    {
        printf("py_ret == nullptr\n");
        return re_str;
    }

    //  拿到返回字符串
    const char* tmp_str = PyUnicode_AsUTF8(py_ret);

    //  检查字符串
    if(tmp_str == nullptr)
    {
        printf("tmp_str == nullptr\n");
        return re_str;
    }

    //  赋值
    re_str = tmp_str;

    //  释放资源
    Py_DecRef(py_ret);
    Py_DecRef(py_prompt);
    //Py_DecRef(py_args);    //  由于其中包含了模型句柄,所以不能释放

    //  操作完成
    return re_str;
}

//  获取已有IR模型的参数
//  成功返回0，其他值表示错误
int CPyAI::GetLocalIRargs(CPyAI::SInferenceArgs* p_args_out)
{
    //  构造完整文件路径
    std::string filepath = CPYAI_EXPORT_IR_MODEL_PATH;
    filepath += "/";
    filepath += CPYAI_MODEL_CONF_FILE;

    //  尝试打开文件
    FILE* fp = fopen(filepath.c_str(), "r");

    //  当文件打开失败
    if(!fp)
    {
        return -1;
    }

    //  读取文件
    SInferenceArgs args;
    fscanf(fp, "%d %d %d", &args.width, &args.height, &args.total_frames);

    //  关闭文件
    fclose(fp);

    //  参数合法性检查
    if((args.width  < 240) || (args.width  > 999) || ((args.width  % 8) != 0)) return -2;
    if((args.height < 240) || (args.height > 999) || ((args.height % 8) != 0)) return -3;
    if((args.total_frames < 16) || (args.total_frames > 999)) return -4;

    //  保存数据
    args.args_valid = true;        //  数据有效
    this->cur_infer_args = args;

    //  同时输出参数到外部
    if(p_args_out != nullptr)
    {
        p_args_out[0] = args;
    }

    //  操作成功
    return 0;
}

//  IR模型是否可用
//  可用返回true,不可用返回false
bool CPyAI::IR_model_is_valid(void)
{
    if(py_ttv_igpunpu_handle != nullptr) return true;
    else                                 return false;
}

//  执行模型转换和编译
void CPyAI::ConvModeltoIRformat_for_iGPU_NPU(CPyAI::SInferenceArgs args)
{
    //  构造输入数据
    PyObject* py_args = PyTuple_New(5);

    //  第一个参数为原始模型
    PyTuple_SetItem(py_args, 0, static_cast<PyObject*>(py_ttv_model_pipe_handle));

    //  第二个参数为输出的IR模型路径
    PyObject* py_output_path = Py_BuildValue("s", CPYAI_EXPORT_IR_MODEL_PATH);
    PyTuple_SetItem(py_args, 1, py_output_path);

    //  第三个参数为视频宽度
    PyObject* py_width = Py_BuildValue("i", args.width);
    PyTuple_SetItem(py_args, 2, py_width);

    //  第四个参数为视频高度
    PyObject* py_height = Py_BuildValue("i", args.height);
    PyTuple_SetItem(py_args, 3, py_height);

    //  第五个参数为视频总帧数
    PyObject* py_frames = Py_BuildValue("i", args.total_frames);
    PyTuple_SetItem(py_args, 4, py_frames);

    //  执行
    PyObject* py_ret = PyObject_CallObject(static_cast<PyObject*>(py_ttv_igpunpy_conv_ir), py_args);

    //  检查
    if(py_ret == nullptr)
    {
        printf("py_ret == nullptr\n");
        return;
    }

    //  释放之前的句柄
    if(py_ttv_igpunpu_handle != nullptr)
    {
        Py_DecRef(static_cast<PyObject*>(py_ttv_igpunpu_handle));
    }

    //  配置
    py_ttv_igpunpu_handle = py_ret;

    //  保存模型参数
    //  构造完整文件路径
    std::string filepath = CPYAI_EXPORT_IR_MODEL_PATH;
    filepath += "/";
    filepath += CPYAI_MODEL_CONF_FILE;

    //  尝试打开文件
    FILE* fp = fopen(filepath.c_str(), "w");

    //  当文件打开失败
    if(!fp)
    {
        printf("[Error] Create IR model config file error!! --- %s\n", filepath.c_str());
        Py_DecRef(py_frames);
        Py_DecRef(py_height);
        Py_DecRef(py_width);
        Py_DecRef(py_output_path);
        return;
    }

    //  写入文件
    fprintf(fp, "%d %d %d\n", args.width, args.height, args.total_frames);

    //  关闭文件
    fclose(fp);

    //  释放资源
    Py_DecRef(py_frames);
    Py_DecRef(py_height);
    Py_DecRef(py_width);
    Py_DecRef(py_output_path);
    //Py_DecRef(py_args);    //  由于其中包含了模型句柄,所以不能释放

    //  操作完成
    return;
}


//  步骤调试
extern void debug_steps_print(int steps);

//  全局变量，用于步骤更新
extern int g_steps;

//  静态回调函数
static PyObject* callback(PyObject* self, PyObject* args)
{
    //  定义变量
    PyObject *p2;
    PyObject *p3;
    long i=0L;
    int tmp = 0;

    //  从参数列表中解析出输入值
    PyArg_ParseTuple(args, "lOO", &i, &p2, &p3);

    //  得到进度值
    tmp = static_cast<int>(i+1);
    g_steps = tmp;
    debug_steps_print(tmp);

    //  返回
    Py_RETURN_NONE;
}

//  执行英文文本到视频文件的生成(使用iGPU+NPU加速器)
void CPyAI::Text_To_Video_with_iGPU_NPU(
    const char* prompt,                  //  输入的英文文本
    int steps,                           //  推理步数
    const char* out_gif_file,            //  输出的gif动图文件
    const char* out_mp4_file             //  输出的mp4视频文件
    )
{
    //  安全检查
    if(py_ttv_igpunpu_handle == nullptr)
    {
        return;
    }

    //  构造输入数据
    PyObject* py_args = PyTuple_New(6);

    //  第一个参数为关键字
    PyObject* py_prompt = Py_BuildValue("s", prompt);
    PyTuple_SetItem(py_args, 0, py_prompt);

    //  第二个参数为IR模型句柄
    PyTuple_SetItem(py_args, 1, static_cast<PyObject*>(py_ttv_igpunpu_handle));

    //  第三个参数为推理步数
    PyObject* py_steps = Py_BuildValue("i", steps);
    PyTuple_SetItem(py_args, 2, py_steps);

    //  第四个参数为输出gif动图文件
    PyObject* py_output_gif_file = Py_BuildValue("s", out_gif_file);
    PyTuple_SetItem(py_args, 3, py_output_gif_file);

    //  第五个参数为输出mp4视频文件
    PyObject* py_output_mp4_file = Py_BuildValue("s", out_mp4_file);
    PyTuple_SetItem(py_args, 4, py_output_mp4_file);

    //  第六个参数为进度回调函数
    PyMethodDef CFunc = {"callback", callback, METH_VARARGS, ""};
    PyObject* pCallbackFunc = PyCFunction_New(&CFunc, nullptr);
    Py_IncRef(pCallbackFunc);
    PyObject* py_progress = Py_BuildValue("O", pCallbackFunc);
    Py_IncRef(py_progress);
    PyTuple_SetItem(py_args, 5, py_progress);

    //  执行
    PyObject_CallObject(static_cast<PyObject*>(py_ttv_igpunpu_ex), py_args);

    //  释放资源
    Py_DecRef(py_progress);
    //Py_DecRef(pCallbackFunc);
    Py_DecRef(py_output_mp4_file);
    Py_DecRef(py_output_gif_file);
    Py_DecRef(py_steps);
    Py_DecRef(py_prompt);
    //Py_DecRef(py_args);    //  由于其中包含了模型句柄,所以不能释放

    //  操作完成
    return;
}

//  执行英文文本到视频文件的生成(未使用加速器加速器)
void CPyAI::Text_To_Video_NoAcc(
    const char* prompt,                  //  输入的英文文本
    int steps,                           //  推理步数
    CPyAI::SInferenceArgs args,          //  推理参数
    const char* out_gif_file,            //  输出的gif动图文件
    const char* out_mp4_file             //  输出的mp4视频文件
    )
{
    //  构造输入数据
    PyObject* py_args = PyTuple_New(9);

    //  第一个参数为关键字
    PyObject* py_prompt = Py_BuildValue("s", prompt);
    PyTuple_SetItem(py_args, 0, py_prompt);

    //  第二个参数为IR模型句柄
    PyTuple_SetItem(py_args, 1, static_cast<PyObject*>(py_ttv_noacc_pipe_handle));

    //  第三个参数为推理步数
    PyObject* py_steps = Py_BuildValue("i", steps);
    PyTuple_SetItem(py_args, 2, py_steps);

    //  第四个参数为视频宽度
    PyObject* py_width = Py_BuildValue("i", args.width);
    PyTuple_SetItem(py_args, 3, py_width);

    //  第五个参数为视频高度
    PyObject* py_height = Py_BuildValue("i", args.height);
    PyTuple_SetItem(py_args, 4, py_height);

    //  第六个参数为总帧数
    PyObject* py_frames = Py_BuildValue("i", args.total_frames);
    PyTuple_SetItem(py_args, 5, py_frames);

    //  第七个参数为输出gif动图文件
    PyObject* py_output_gif_file = Py_BuildValue("s", out_gif_file);
    PyTuple_SetItem(py_args, 6, py_output_gif_file);

    //  第八个参数为输出mp4视频文件
    PyObject* py_output_mp4_file = Py_BuildValue("s", out_mp4_file);
    PyTuple_SetItem(py_args, 7, py_output_mp4_file);

    //  第九个参数为进度回调函数
    PyMethodDef CFunc = {"callback", callback, METH_VARARGS, ""};
    PyObject* pCallbackFunc = PyCFunction_New(&CFunc, nullptr);
    PyObject* py_progress = Py_BuildValue("O", pCallbackFunc);
    PyTuple_SetItem(py_args, 8, py_progress);

    //  执行
    PyObject_CallObject(static_cast<PyObject*>(py_ttv_noacc_ex), py_args);

    //  释放资源
    Py_DecRef(py_progress);
    Py_DecRef(py_output_mp4_file);
    Py_DecRef(py_output_gif_file);
    Py_DecRef(py_frames);
    Py_DecRef(py_height);
    Py_DecRef(py_width);
    Py_DecRef(py_steps);
    Py_DecRef(py_prompt);
    //Py_DecRef(py_args);    //  由于其中包含了模型句柄,所以不能释放

    //  操作完成
    return;
}
