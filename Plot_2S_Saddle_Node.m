% clear all
% load Prediction_out_r2.txt
% 
% start_u = 0.08;
% end_u = 0.75;
% 
% steps = 501;
% us = linspace(start_u, end_u, steps);
% 
% for n=5000:length(Prediction_out_r2(:,1))
%     plot(Prediction_out_r2(n,:), us)
%     pause(0.01)
% end

clear all
% load Prediction_out_a2.txt
load A_2S_2nd_gam_eq.txt
load A_2S_2nd_u_train.txt
load A_2S_2nd_v_eq.txt

start_u = A_2S_2nd_u_train(1);
end_u = A_2S_2nd_u_train(end);
% 
steps = length(A_2S_2nd_u_train);
us = linspace(start_u, end_u, steps);
hold on
plot(A_2S_2nd_gam_eq, A_2S_2nd_u_train)
plot(A_2S_2nd_gam_eq, A_2S_2nd_v_eq)
% 