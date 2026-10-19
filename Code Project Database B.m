clear;
close all;
clc;

%% CARICAMENTO DATI E INIZIALIZZAZIONE

load('databaseB.mat');
data = HRV.series;

ns = size(data,1);  %numero soggetti
fasi_idx = [1 2 4];
fasi = {'REST','TILT','MENTAL ARITHMETIC'};
nf = length(fasi_idx); %numero fasi selez.
alpha = 0.05; %livello di significativita

pat_mean = nan(ns,nf);
pat_sd = nan(ns,nf);

corr_rp = nan(ns,nf);
slope_rp = nan(ns,nf);
offset_rp = nan(ns,nf);
r2_rp = nan(ns,nf);

gc_rp = nan(ns,nf);
gc_pr = nan(ns,nf);
p_gc_rp = nan(ns,nf);
p_gc_pr = nan(ns,nf);
ord_sel = nan(ns,nf);

%% COSTRUZIONE E VISUALIZZAZIONE DI RRI E PPI

sub = 1;

figure(1);

for i = 1:nf

    s = data{sub,fasi_idx(i)};

    rr = 1000*diff(s(:,1));
    pp = 1000*diff(s(:,2));

    subplot(nf,1,i);
    plot(rr,'LineWidth',1);
    hold on;
    plot(pp,'LineWidth',1);
    grid on;
    xlabel('Intervallo');
    ylabel('Durata (ms)');
    title(fasi{i});
    legend('RRI','PPI','Location','best');

end


sgtitle(['Serie RRI e PPI - soggetto ' num2str(sub)]);

%% COSTRUZIONE E VISUALIZZAZIONE DEL PAT

figure(2);

for i = 1:nf

    s = data{sub,fasi_idx(i)};

    pat = 1000*(s(:,2)-s(:,1));

    subplot(nf,1,i);
    plot(pat,'LineWidth',1);
    grid on;
    xlabel('Battito');
    ylabel('PAT (ms)');
    title(fasi{i});

end


sgtitle(['Serie del PAT - soggetto ' num2str(sub)]);

%% PAT MEDIO E DEVIAZIONE STANDARD

for i = 1:nf

    for n = 1:ns

        s = data{n,fasi_idx(i)};

        pat = 1000*(s(:,2)-s(:,1));

        pat_mean(n,i) = mean(pat);
        pat_sd(n,i) = std(pat);

    end

end


figure(3);

subplot(1,2,1);
boxplot(pat_mean,'Labels',fasi);
ylabel('PAT medio (ms)');
title('PAT medio nelle tre condizioni');
grid on;

subplot(1,2,2);
boxplot(pat_sd,'Labels',fasi);
ylabel('Deviazione standard del PAT (ms)');
title('Variabilita del PAT');
grid on;

%% CORRELAZIONE E REGRESSIONE LINEARE

for i = 1:nf

    for n = 1:ns

        s = data{n,fasi_idx(i)};

        rr = 1000*diff(s(:,1));
        pp = 1000*diff(s(:,2));

        corr_rp(n,i) = corr(rr,pp);

        coeff = polyfit(rr,pp,1);
        slope_rp(n,i) = coeff(1);
        offset_rp(n,i) = coeff(2);

        pp_pred = polyval(coeff,rr);
        rss = sum((pp-pp_pred).^2);
        tss = sum((pp-mean(pp)).^2);
        r2_rp(n,i) = 1-rss/tss;

    end

end


%% RISULTATI CORRELAZIONE E REGRESSIONE

fprintf('\n===== CORRELAZIONE E REGRESSIONE RRI-PPI =====\n');

for i = 1:nf

    fprintf('%s | r mediano = %.3f | pendenza media = %.3f | intercetta media = %.3f ms | R2 medio = %.3f\n',fasi{i},median(corr_rp(:,i)),mean(slope_rp(:,i)),mean(offset_rp(:,i)),mean(r2_rp(:,i)));

end


%% GRAFICI DELLA REGRESSIONE LINEARE

figure(4);

for i = 1:nf

    s = data{sub,fasi_idx(i)};

    rr = 1000*diff(s(:,1));
    pp = 1000*diff(s(:,2));

    coeff = polyfit(rr,pp,1);

    x_reg = linspace(min(rr),max(rr),100);
    y_reg = polyval(coeff,x_reg);

    subplot(1,nf,i);
    scatter(rr,pp,20,'filled');
    hold on;
    plot(x_reg,y_reg,'r-','LineWidth',1.5);
    grid on;
    xlabel('RRI (ms)');
    ylabel('PPI (ms)');
    title(fasi{i});
    legend('Dati','Regressione','Location','best');

end


sgtitle(['Regressione lineare RRI-PPI - soggetto ' num2str(sub)]);

%% PREPROCESSING E CAUSALITA DI GRANGER

for i = 1:nf

    for n = 1:ns

        s = data{n,fasi_idx(i)};

        rr = 1000*diff(s(:,1));
        pp = 1000*diff(s(:,2));

        %% PREPROCESSING

        det_ord = 8;

        rr = detrend(rr,det_ord);
        rr = rr-mean(rr);

        pp = detrend(pp,det_ord);
        pp = pp-mean(pp);

        z = [rr pp];

        %% SCELTA DELL'ORDINE CON AKAIKE

        ord_max = 12;
        ord_out = VARorder(z,ord_max);
        ord = ord_out.pottaic;
        ord_sel(n,i) = ord;

        %% MODELLI PER RRI -> PPI

        ar_pp = LinReg(z,2,2,1:ord);
        arx_pp = LinReg(z,2,[1 2],1:ord);

        %% MODELLI PER PPI -> RRI

        ar_rr = LinReg(z,1,1,1:ord);
        arx_rr = LinReg(z,1,[1 2],1:ord);

        %% MISURE DI GRANGER

        gc_rp(n,i) = log(ar_pp.es2u/arx_pp.es2u);
        gc_pr(n,i) = log(ar_rr.es2u/arx_rr.es2u);

        %% F-TEST

        nb_full = 2*ord;
        nb_red = ord;

        [p_gc_rp(n,i),~] = LinReg_Ftest(arx_pp.eu,ar_pp.eu,nb_full,nb_red);
        [p_gc_pr(n,i),~] = LinReg_Ftest(arx_rr.eu,ar_rr.eu,nb_full,nb_red);

    end

end


%% PERCENTUALE DI SOGGETTI SIGNIFICATIVI

perc_rp = sum(p_gc_rp < alpha)/ns*100;
perc_pr = sum(p_gc_pr < alpha)/ns*100;

bar_data = [perc_rp; perc_pr]';

figure(5);
bar(bar_data);
ylim([0 110]);
xticks(1:nf);
xticklabels(fasi);
ylabel('Percentuale dei soggetti (%)');
title('Significativita della causalita di Granger');
legend('RRI -> PPI','PPI -> RRI','Location','best');
grid on;

%% ANOVA TRA LE FASI

[p_anova_rp,tbl_rp] = anova1(gc_rp,[],'off');
[p_anova_pr,tbl_pr] = anova1(gc_pr,[],'off');

disp(['ANOVA RRI -> PPI, p-value = ' num2str(p_anova_rp)]);
disp(['ANOVA PPI -> RRI, p-value = ' num2str(p_anova_pr)]);

fprintf('\n');

%% T-TEST TRA LE FASI

h_rp = nan(nf,nf);
p_rp = nan(nf,nf);
h_pr = nan(nf,nf);
p_pr = nan(nf,nf);

for i = 1:nf

    for j = i+1:nf

        [h_rp(i,j),p_rp(i,j)] = ttest(gc_rp(:,i),gc_rp(:,j),'Alpha',alpha);
        [h_pr(i,j),p_pr(i,j)] = ttest(gc_pr(:,i),gc_pr(:,j),'Alpha',alpha);

    end

end


disp('P-value tra le fasi per RRI -> PPI');
disp(p_rp);

disp('P-value tra le fasi per PPI -> RRI');
disp(p_pr);

%% T-TEST TRA LE DIREZIONI NELLA STESSA FASE

h_dir = nan(1,nf);
p_dir = nan(1,nf);

for i = 1:nf

    [h_dir(i),p_dir(i)] = ttest(gc_rp(:,i),gc_pr(:,i),'Alpha',alpha);

end


disp('P-value del confronto RRI -> PPI contro PPI -> RRI');
disp(p_dir);

%% LIMITI DEI GRAFICI

ymax = 1.15*max([max(gc_rp(:)) max(gc_pr(:))]);
ymin = 0;
dy = (ymax-ymin)/100;

%% BOXPLOT DELLE DUE DIREZIONI

figure(6);

subplot(1,2,1);
boxplot(gc_rp,'Widths',0.3,'Colors',[0 0.5 1],'Symbol','.','Positions',1:nf);
xticklabels(fasi);
ylabel('Causalita di Granger');
title('RRI -> PPI');
ylim([ymin ymax]);
grid on;

if p_anova_rp < alpha

    for i = 1:nf

        for j = i+1:nf

            if h_rp(i,j) == 1

                py = ymax-(i+j)*dy;
                line([i j],[py py],'Color',[0.75 0.5 0],'LineStyle','--');
                text((i+j)/2,py+dy,'*','HorizontalAlignment','center','Color',[0.75 0.5 0],'FontSize',16);

            end

        end

    end

end


subplot(1,2,2);
boxplot(gc_pr,'Widths',0.3,'Colors',[0.2 0.8 0.2],'Symbol','.','Positions',1:nf);
xticklabels(fasi);
ylabel('Causalita di Granger');
title('PPI -> RRI');
ylim([ymin ymax]);
grid on;

if p_anova_pr < alpha

    for i = 1:nf

        for j = i+1:nf

            if h_pr(i,j) == 1

                py = ymax-(i+j)*dy;
                line([i j],[py py],'Color',[0.75 0.5 0],'LineStyle','--');
                text((i+j)/2,py+dy,'*','HorizontalAlignment','center','Color',[0.75 0.5 0],'FontSize',16);

            end

        end

    end

end


%% CONFRONTO TRA LE DIREZIONI

figure(7);

boxplot(gc_rp,'Widths',0.3,'Colors',[0 0.5 1],'Symbol','.','Positions',(1:nf)-0.2);
hold on;
boxplot(gc_pr,'Widths',0.3,'Colors',[0.2 0.8 0.2],'Symbol','.','Positions',(1:nf)+0.2);

xlim([0.5 nf+0.5]);
ylim([ymin ymax]);
xticks(1:nf);
xticklabels(fasi);
ylabel('Causalita di Granger');
title('RRI -> PPI contro PPI -> RRI');
grid on;

py = ymax-5*dy;

for i = 1:nf

    if h_dir(i) == 1

        line([i-0.2 i+0.2],[py py],'Color',[0.75 0.5 0],'LineStyle','--');
        text(i,py+dy,'*','HorizontalAlignment','center','Color',[0.75 0.5 0],'FontSize',16);

    end

end
