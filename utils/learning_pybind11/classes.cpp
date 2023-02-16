/**
 * @Description: 
 * @version: v1.0
 * @Author: HTY
 * @Date: 2023-01-21 23:54:40
 */

#include <pybind11/pybind11.h>
#include <iostream>

class Pet
{
public:
    enum Kind {
        Dog = 0,
        Cat
    } type;

    Pet(const std::string &name, Kind type) : name(name), type(type) { }
    void setName(const std::string &name_) { name = name_; }
    const std::string &getName() const { return name; }
    void set(int age_) {age = age_; }
    void set(const std::string &sex_) { sex = sex_; } // overload
    std::string info()
    {
        return "name: " + name + "\nage: " + std::to_string(age) + "\nsex: " + sex + "\n";
    }

private:
    std::string name;
    int age;
    std::string sex;
};

class Dog : public Pet
{
public:
    explicit Dog(const std::string &name) : Pet(name, Pet::Dog) { }
    std::string bark() const { return "Woof!"; }
};

PYBIND11_MODULE(classes, m)
{
    pybind11::class_<Pet> pet(m, "Pet", pybind11::dynamic_attr());  // 允许在Python中直接添加类的属性
    pet.def(pybind11::init<const std::string &, Pet::Kind>());
    pet.def_property("name", &Pet::getName, &Pet::setName);
    pet.def_readwrite("type", &Pet::type);
    pet.def("info", &Pet::info);

    // use enum
    pybind11::enum_<Pet::Kind>(pet, "Kind")
            .value("Dog", Pet::Kind::Dog)
            .value("Cat", Pet::Kind::Cat)
            .export_values();

    // two ways of indicating an overload function
    pet.def("set", static_cast<void (Pet::*)(int)>(&Pet::set), "Set the pet's age");
    pet.def("set", static_cast<void (Pet::*)(const std::string &)>(&Pet::set), "Set the pet's sex");
    // if using C++14, identical to
//    pet.def("set", pybind11::overload_cast<int>(&Pet::set), "Set the pet's age");
//    pet.def("set", pybind11::overload_cast<const std::string &>(&Pet::set), "Set the pet's sex");

    // two ways of indicating a hierarchical relationship
    pybind11::class_<Dog, Pet>(m, "Dog")    // <- specify C++ parent type
            .def(pybind11::init<const std::string &>())
            .def("bark", &Dog::bark);
    // identical to
//    pybind11::class_<Dog>(m, "Dog", pet)    // <- specify Python parent type
//            .def(pybind11::init<const std::string &>())
//            .def("bark", &Dog::bark);
}


