function xdot = dubins_car(t, x, u)
% system dynamics
xdot = [x(3)*cos(x(4)); x(3)*sin(x(4)); u(1); u(2)];
end