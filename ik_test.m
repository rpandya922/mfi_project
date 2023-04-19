dhparams =[0,0,0.050,-1.5708;
           -1.5708,0,0.440,3.1416;
           0,0,0.035,-1.5708;
           0,-0.420,0,1.5708;
           0,0,0,-1.5708;
           0,-0.080-0.23,0,3.1416];%theta,d,a,alpha gripper offset 0.23m.
       
% order for matlab setFixedTransform: [a alpha d theta]
DH = [dhparams(:,3) dhparams(:,4) dhparams(:,2) dhparams(:,1)];

robot_tree = rigidBodyTree;
body1 = rigidBody('body1');
jnt1 = rigidBodyJoint('jnt1','revolute');
setFixedTransform(jnt1, DH(1,:),'dh');
body1.Joint = jnt1;
addBody(robot_tree, body1, 'base');

body2 = rigidBody('body2');
jnt2 = rigidBodyJoint('jnt2', 'revolute');
body3 = rigidBody('body3');
jnt3 = rigidBodyJoint('jnt3', 'revolute');
body4 = rigidBody('body4');
jnt4 = rigidBodyJoint('jnt4', 'revolute');
body5 = rigidBody('body5');
jnt5 = rigidBodyJoint('jnt5', 'revolute');
body6 = rigidBody('body6');
jnt6 = rigidBodyJoint('jnt6', 'revolute');

setFixedTransform(jnt2, DH(2,:),'dh');
setFixedTransform(jnt3, DH(3,:),'dh');
setFixedTransform(jnt4, DH(4,:),'dh');
setFixedTransform(jnt5, DH(5,:),'dh');
setFixedTransform(jnt6, DH(6,:),'dh');

body2.Joint = jnt2;
body3.Joint = jnt3;
body4.Joint = jnt4;
body5.Joint = jnt5;
body6.Joint = jnt6;

addBody(robot_tree, body2, 'body1');
addBody(robot_tree, body3, 'body2');
addBody(robot_tree, body4, 'body3');
addBody(robot_tree, body5, 'body4');
addBody(robot_tree, body6, 'body5');

robot_tree.DataFormat = 'column';

% try creating IK solver
ik = inverseKinematics('RigidBodyTree', robot_tree);
weights = [0.25 0.25 0.25 1 1 1];
init_guess = [0; 0; 0; 0; 0; 0];
pose = eye(4);
pose(1:3,4) = [0.5; 0.5; 0.2];
pose(1:3,1:3) = [  0.0000000,  0.0000000,  1.0000000;
   0.0000000,  1.0000000,  0.0000000;
  -1.0000000,  0.0000000,  0.0000000 ];
% pose(1:3,1:3) = [ -1.0000000,  0.0000000,  0.0000000;
%    0.0000000,  1.0000000,  0.0000000;
%   -0.0000000,  0.0000000, -1.0000000 ];
t0 = tic;
[configSol, solInfo] = ik('body6', pose, weights, init_guess);
t1 = toc(t0);
disp("solve took " + t1 + " seconds");
show(robot_tree, configSol);
