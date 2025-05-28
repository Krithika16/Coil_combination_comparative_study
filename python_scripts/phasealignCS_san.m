% Credit: San Xiang wrote this code

function [IMPA_mW,IMPA_wPhase, IMPA] = phasealignCS_san(compdata)%181011 san's coil combo method
    
    [nr,nc,npc,nsl,nch] = size(compdata); %K - extract the shape of the image data
    for sl = 1:nsl %run this algorithm on each slice
        
        cdata = squeeze(compdata(:,:,:,sl,:));%cdata size is (nr,nc,(K)npc,nch) - only removing the slice dimension by squeezing
        %cdata(1:5,1,1,1)
        %% 1. commpute Complex Sum for each channel over all phase cycles for reference
        temp = mean(cdata,3);
        %temp(1:5, 1,1,1)
        %size(temp)
        CSref = squeeze(mean(cdata,3));%complex sum over the phase cycles for each channel - CSref size is (nr, nc, nch)
        %CSref(1:5,1,1)
        %size(CSref)
        superCS = mean(CSref,3);%now finding the complex average sum across the channels - size should be (nr, nc) - this is denominator of the weights for each channel
        phasorCSref = CSref./abs(CSref); % unit vector for CSref - This is what is used for phase alignment lol - what is complex signal divided by magnitude signal
        phasorSuperCS = superCS./abs(superCS);  % this is used to get back the phase we remove during alignment - not needed as I simply want to focus on coil combination
        %     figure;%for debugging - this shows the magnitude and phase images
    %     for the complex sum images for each channel
    %     for ch = 1:nch
    %         subplot(3,5,ch);imagesc(abs(CSref(:,:,ch)));colormap(gray);title(['Coil#' num2str(ch)]); axis image
    %     end
    %     [~,h1]=suplabel('Complex Sum (over Phase Cycles) Mag','t');
    %     if pr, print(gcf,'-djpeg100','PC-CS_Mag');saveas(gcf,'PC-CS_Mag'); end    
    %     
    %     figure;%for debugging
    %     for ch = 1:nch
    %         subplot(3,5,ch);imagesc(angle(CSref(:,:,ch)));colormap(gray);title(['Coil#' num2str(ch)]); axis image
    %     end
    %     [~,h1]=suplabel('Complex Sum (over Phase Cycles) Phase','t');
    %     if pr, print(gcf,'-djpeg100','PC-CS_Phase');saveas(gcf,'PC-CS_Phase'); 
    
    
%% 2. remove CS-reference phase from each image for all phase cycles
    %conj(phasorCSref(1:10,1,1))
    for ch = 1:nch %Loops over each channel
        for pc = 1:npc %loops over each phase cycle
            %k - in the below code, the complex conjugate of the unit vector of the complex average sum is
            %multiplied elementwise with the image from each phase cycle
            %for a particular channel. This is likely done to align the
            %phases as some of the phase will be removed when multiplying
            %with this conjugate
            
            compReffedCS(:,:,pc,ch) = conj(phasorCSref(:,:,ch)).*cdata(:,:,pc,ch);
        end
    end
%     for pc = 1:npc
%         figure;%for debugging
%         for ch = 1:nch
%             subplot(3,5,ch);imagesc(angle(compReffedCS(:,:,pc,ch)));title(['Coil#' num2str(ch)]); colormap(gray); axis image
%         end
%         str = ['PC-CS Referenced Phase, ' num2str((pc-1)*90) '{\circ}'];[~,h1]=suplabel(str,'t');
%         if pr, str = ['PC-CSref_Phase_PC=' num2str((pc-1)*90)];print(gcf,'-djpeg100',str);saveas(gcf,str); end    
%     end

%% 3. form Magnitude-weighted combo
    for pc = 1:npc %Do coil combination for each phase cycle 
        IMPA_mW(:,:,pc,sl) = sum(abs(compReffedCS(:,:,pc,:)).*compReffedCS(:,:,pc,:)./sum(abs(compReffedCS(:,:,pc,:)),4),4);% a magnitude-weighted IMPA - not sure what this is used for...
        IMPA(:,:,pc,sl) = sum(abs(CSref).*squeeze(compReffedCS(:,:,pc,:))./sum(abs(CSref),3),3);%IMPA according to San - this is the formula given in the IMPA - this is what I need to use.
%         figure(100);subplot(1,4,pc); imagesc(abs(compcombo(:,:,pc,sl)));colormap(gray); title([num2str((pc-1)*90) '{\circ} PC']); axis image
    end
%     [~,h1]=suplabel('PC-CS Referenced Combo Mag','t');
%     if pr, str = 'PC-CSref_ComboMag';print(gcf,'-djpeg100',str);saveas(gcf,str); end
%     
%     for pc = 1:npc
%         figure(101);subplot(1,4,pc); imagesc(angle(compcombo(:,:,pc,sl)));colormap(gray);title([num2str((pc-1)*90) '{\circ} PC']); axis image
%     end
%     [~,h1]=suplabel('PC-CS Referenced Combo Phase','t');
%     if pr, str = 'PC-CSref_ComboPhase';print(gcf,'-djpeg100',str);saveas(gcf,str); end
%     
%     for pc = 1:npc
%         figure(102);subplot(1,4,pc); imagesc(abs(compCScombo(:,:,pc,sl)));colormap(gray);title([num2str((pc-1)*90) '{\circ} PC']); axis image
%     end
%     [~,h1]=suplabel('PC-CS Referenced CSw Combo Mag','t');
%     if pr, str = 'PC-CSref_CSwComboMag';print(gcf,'-djpeg100',str);saveas(gcf,str); end
%     
%     for pc = 1:npc
%         figure(103);subplot(1,4,pc); imagesc(angle(compCScombo(:,:,pc,sl)));colormap(gray);title([num2str((pc-1)*90) '{\circ} PC']); axis image
%     end
%     [~,h1]=suplabel('PC-CS Referenced CSw Combo Phase','t');
%     if pr, str = 'PC-CSref_CSwComboPhase';print(gcf,'-djpeg100',str);saveas(gcf,str); end
   %% try to get the CS solution phase back by reapplying the CS phase that as originally removed - don't think i need this.a all i really care about it is coil comb lol
    for pc = 1:npc
%         compCSphased(:,:,pc) = conj(expSuperCS(:,:,sl)).*compCScombo(:,:,pc,sl);
        IMPA_wPhase(:,:,pc, sl) = phasorSuperCS(:,:,sl).*IMPA(:,:,pc,sl);
%        figure(104);subplot(1,4,pc); imagesc(angle(compCSphased(:,:,pc,sl)));colormap(gray);title([num2str((pc-1)*90) '{\circ} PC']); axis image
    end
%     [~,h1]=suplabel('PC-CS Referenced CSw Combo Phase, Rephased','t');
%     if pr, str = 'PC-CSref_CSwComboRePhased';print(gcf,'-djpeg100',str);saveas(gcf,str); end
    end
end