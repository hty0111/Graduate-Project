/**
 * @Description: 
 * @version: v1.0
 * @Author: HTY
 * @Date: 2023-01-21 15:38:54
 */

#include <pybind11/pybind11.h>

int add(int i = 1, int j = 2)
{
    return i + j;
}

PYBIND11_MODULE(functions, m)	// 模块名, 模块实例对象
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    // 特定名称的参数和默认参数
    m.def("add", &add, "A function that adds two numbers", pybind11::arg("i") = 1, pybind11::arg("j") = 2);
    // identical to
//    using namespace pybind11::literals;
//    m.def("add", &add, "i"_a=1, "j"_a=2);
}