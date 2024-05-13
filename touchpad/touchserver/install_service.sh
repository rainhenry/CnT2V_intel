#!/bin/bash
sudo cp touchpad.service /etc/systemd/user/touchpad.service 
systemctl --user daemon-reload
systemctl --user enable touchpad.service

sudo cp 99-touchpad.rules /etc/udev/rules.d/

sudo cp touchserver /usr/bin

