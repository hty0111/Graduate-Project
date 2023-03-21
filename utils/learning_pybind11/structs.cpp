/**
 * @Description: 
 * @version: v1.0
 * @Author: HTY
 * @Date: 2023-01-21 23:22:48
 */


#include <pybind11/pybind11.h>
#include <iostream>

struct People
{
    explicit People(const std::string &name="HTY") : name(name) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }

    std::string name;
    const int age = 22;
};


PYBIND11_MODULE(structs, m)
{
    pybind11::class_<People>(m, "People")
            .def(pybind11::init<const std::string &>(), pybind11::arg("name") = "HTY")
            .def("setName", &People::setName)
            .def("getName", &People::getName)
            .def_readwrite("name", &People::name)   // 读写变量
            .def_readonly("age", &People::age)      // 只读变量，const类型
            .def("__repr__", [](const People &a) {  // 用lambda函数的方式重载print()函数
                return "People named " + a.name;
            });
}


