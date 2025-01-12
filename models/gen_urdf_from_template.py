from jinja2 import Template

arm_length = 0.15
arm_length_geom = 0.115
com_from_vehicle = 0.08
com_from_ee = (arm_length + 0.05) - com_from_vehicle
output_name = "uam_0dof_com_far.urdf"

with open("uam_0dof_template.urdf", "r") as file:
    template = Template(file.read())

params = {
    "arm_length": arm_length,
    "arm_length_geom": arm_length_geom,
    "com_from_vehicle": com_from_vehicle,
    "com_from_ee": com_from_ee
}

output = template.render(params)

with open(output_name, "w") as file:
    file.write(output)