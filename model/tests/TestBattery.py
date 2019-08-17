# 11 Mar, 2018

from Battery import Battery

b = Battery()
print(b.get_current_battery_percentage()) # should be 100.0
b.update_battery_percentage(b.get_battery_capacity()/2)
print(b.get_current_battery_percentage()) # should be 50.0
b.set_current_battery_percentage(25.0)
print(b.get_current_battery_percentage()) # should be 25.0
