clc; clear all; close all;

cd D:\27_3_2024_HV_T2mapp;

% Credit: San Xiang wrote this code

load('ax_moved_ss_allpc.mat')

% figure;
% imagesc(abs(dataIm_pre_phaseAlign(:,:,1,1,1)));
% colormap("gray")

[IMPA_mW,IMPA_wPhase, IMPA] = phasealignCS_san(ax_moved_ss_allpc);

[nr, nc, npc] = size(IMPA);

for pc = 1:npc
    
    figure();
    imagesc(angle(IMPA(:,:,pc)));
    colormap("gray");
    title(["Phase cycle: " num2str(pc)])
end
%% 
% figure(pc+1);
% for pc = 1:npc
%     plot(real(IMPA(120, 230, pc)), imag(IMPA(120, 230, pc)))
% 
% end

%complex_image = IMPA_mW + (1i*IMPA_wPhase);

for pc = 1:npc
    figure()
    imagesc(angle(IMPA_wPhase(:,:,pc)));
    colormap("gray");
    title(["Phase cycle: " num2str(pc)])
end

save IMPA_27_3_2024.mat IMPA IMPA_wPhase IMPA_mW;