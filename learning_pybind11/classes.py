import classes

# print(help(classes.Pet))

p = classes.Pet("Charly", classes.Pet.Dog)
p.color = "white"   # dynamic attribute
print(p.__dict__)
p.set(5)    # set age
p.set("male")   # set sex
print(p.info())

d = classes.Dog("Molly")
print(d.bark())
