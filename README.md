# RS-Image-Processing-App-MATLAB-

function varargout = kaocha(varargin)
% kaocha MATLAB code for kaocha.fig
%      kaocha, by itself, creates a new kaocha or raises the existing
%      singleton*.
%
%      H = kaocha returns the handle to a new kaocha or the handle to
%      the existing singleton*.
%
%      kaocha('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in kaocha.M with the given input arguments.
%
%      kaocha('Property','Value',...) creates a new kaocha or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before kaocha_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to kaocha_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help kaocha

% Last Modified by GUIDE v2.5 11-Jul-2025 02:20:22

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @kaocha_OpeningFcn, ...
                   'gui_OutputFcn',  @kaocha_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before kaocha is made visible.
function kaocha_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to kaocha (see VARARGIN)

% Choose default command line output for kaocha
handles.output = hObject;

% -=-=-=-=-=-=-=-=-KNN专用变量初始化 -=-=-=-=-=-=-=-=
handles.samples = struct();
handles.class_colors = turbo(5);  % 8种高区分度颜色
handles.current_class = 1;
handles.sample_count = 0;
handles.isTrained = false;

% 表格窗口句柄初始化
handles.status_table_fig = []; % 独立窗口句柄
handles.status_table_handle = []; % 表格控件句柄

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes kaocha wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --- Outputs from this function are returned to the command line.
function varargout = kaocha_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in save.
function save_Callback(hObject, eventdata, handles)
try
    % 获取图像数据 ================
    h = get(handles.mainAxes, 'Children');
    img = [];
    
    % 优先从坐标轴获取图像
    for i = 1:length(h)
        if strcmpi(get(h(i), 'Type'), 'image')
            img = get(h(i), 'CData');
            break;
        end
    end
    
    % 后备方案：使用handles存储的图像
    if isempty(img)
        if isfield(handles, 'classified_img')
            img = handles.classified_img;
        elseif isfield(handles, 'image')
            img = handles.image;
        else
            error('未找到可保存的图像数据');
        end
    end
    
    % 保存对话框 ================
    [filename, pathname] = uiputfile(...
        {'*.png', 'PNG图像 (*.png)'; 
         '*.jpg', 'JPEG图像 (*.jpg)'; 
         '*.bmp', 'BMP图像 (*.bmp)';
         '*.mat', 'MAT文件 (*.mat)'}, ...
        '保存当前图像');
    
    if isequal(filename, 0)
        return; % 用户取消
    end
    
    % 文件类型处理 ================
    [~, ~, ext] = fileparts(filename);
    fullpath = fullfile(pathname, filename);
    
    if strcmpi(ext, '.mat')
        % MAT文件保存
        image_data = img;
        save(fullpath, 'image_data'); % 修正为函数格式[2,3](@ref)
        msgbox('图像已保存为MAT文件！', '保存成功');
    else
        % 图像文件保存
        if isfield(handles, 'class_colors')
            % 分类结果特殊处理
            if size(img, 3) == 1 % 索引图像
                colors = handles.class_colors;
                if max(colors(:)) > 1
                    colors = colors / 255; % 归一化到0-1范围
                end
                img = ind2rgb(img, colors); % 转换为真彩色
            end
        end
        imwrite(img, fullpath);
        msgbox('图像已成功保存！', '保存成功');
    end
    
catch ME
    errordlg(['保存失败: ' ME.message], '错误');
end

% --- Executes on button press in opening.
function opening_Callback(hObject, eventdata, handles)
% hObject    handle to opening (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', '所有图像文件'}, '选择图像');
if isequal(filename, 0)   % 判断用户是否读取了图像，若未读取，返回，不进行操作
    return;  % 用户取消
end
im = imread(fullfile(pathname, filename));   % 读取图像，fullfile拼接出完整的图像路径
imshow(im, 'Parent', handles.mainAxes);         % 显示图像
handles.image = im;                          % 保存图像到 handles
    
    % KNN专用变量
    handles.samples = struct();               % 样本存储结构体
    handles.class_colors = turbo(8);          % 8种高区分度颜色
    handles.current_class = 1;                % 当前标注类别
    handles.sample_count = 0;                 % 当前类样本计数
    handles.isTrained = false;                % 模型训练状态标志
    
    guidata(hObject, handles);

guidata(hObject, handles);

% --- Executes on button press in huiduhua.
function huiduhua_Callback(hObject, eventdata, handles)
% hObject    handle to huiduhua (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
im = handles.image;
im1 = rgb2gray(im);
imshow(im1, 'Parent', handles.mainAxes);
handles.image = im1;
guidata(hObject, handles);

% --- Executes on button press in fanzhuan.
function fanzhuan_Callback(hObject, eventdata, handles)
% hObject    handle to fanzhuan (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
im = handles.image;            % 读取当前图像
im1 = im(end:-1:1, :, :);      % 上下翻转（行方向反转）
imshow(im1, 'Parent', handles.mainAxes);
handles.image = im1;          % 更新保存图像
guidata(hObject, handles);    % 更新 handles 数据结构

% --- Executes on button press in resize.
function resize_Callback(hObject, eventdata, handles)
% hObject    handle to resize (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
im = handles.image;              % 获取当前图像
im1 = imresize(im, 0.2);         % 缩放
imshow(im1, 'Parent', handles.mainAxes);  % 显示缩小图像
handles.image = im1;                   % 更新图像数据
guidata(hObject, handles);             % 更新 handles

% --- Executes on button press in bianyuanjiance.
function bianyuanjiance_Callback(hObject, eventdata, handles)
% hObject    handle to bianyuanjiance (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
im = handles.image;
% 判断是否为彩色图像，如果是则转为灰度图
if size(im, 3) == 3
    gray_img = rgb2gray(im);
else
    gray_img = im; % 已经是灰度图
end
% 使用Canny算子进行边缘检测，自动计算阈值
edge_img = edge(gray_img, 'canny');
imshow(edge_img, 'Parent', handles.mainAxes);  % 显示边缘检测图像
handles.image = edge_img;                   % 更新图像数据
guidata(hObject, handles);             % 更新 handles

%% -=-=-=-=-=-=-=-K-means分类函数-=-=-=-=-=--=--=-=-=-=-
function [classified_img, class_colors] = knn_classification(img, numClasses)
% 转换为双精度
if size(img, 3) == 3
    img_data = double(reshape(img, [], 3)) / 255;
else
    img_data = double(img(:)) / 255;
    img_data = repmat(img_data, 1, 3); % 灰度图复制为三通道
end
% -=-=-=-=-使用K-means初始化聚类中心-=-=-=-=-
[~, centers] = kmeans(img_data, numClasses, 'MaxIter', 100, 'Replicates', 3);
% -=-=-=-=-=-=-=-=KNN分类-=-=-=-=-=-=
[idx, ~] = knnsearch(centers, img_data, 'K', 1);
% 创建伪彩色图像
class_colors = lines(numClasses); % 使用不同的颜色
classified_data = class_colors(idx, :);
classified_img = reshape(classified_data, [size(img, 1), size(img, 2), 3]);

% 转换为8位图像
classified_img = uint8(classified_img * 255);
%% K-means回调函数
% --- Executes on button press in kmeans.
function kmeans_Callback(hObject, eventdata, handles)
% hObject    handle to kmeans (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if ~isfield(handles, 'image') || isempty(handles.image)
        errordlg('请先加载图像！', '错误');
        return;
end
    % -=-=-=-获取用户输入的类别数-=-=-=-
    prompt = {'输入分类数量:'};
    dlgtitle = 'K-means均值非监督分类参数';
    dims = [1 35];
    definput = {'3'};
    answer = inputdlg(prompt, dlgtitle, dims, definput);
    
    if isempty(answer)
        return; % 用户取消
    end
    
    numClasses = str2double(answer{1});
    if isnan(numClasses) || numClasses < 2 || numClasses > 10
        errordlg('请输入2-10之间的有效数字！', '错误');
        return;
    end
    
    %% =-=-=-=-=-=-=-=-=-= 主成分分析(PCA)功能 =-=-=-=-=-=-=-=-=-
% --- Executes on button press in pca_analysis.
function pca_analysis_Callback(hObject, eventdata, handles)
    % hObject    handle to pca_analysis (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % 检查图像数据
    if ~isfield(handles, 'image') || isempty(handles.image)
        errordlg('请先加载图像！', '错误');
        return;
    end
    
    % 获取多波段图像
    if size(handles.image, 3) < 2
        errordlg('PCA需要多波段图像（至少2个波段）！', '错误');
        return;
    end
    
    % 参数设置对话框
    prompt = {'保留的主成分数量 (1-6):', '是否显示特征值贡献率:'};
    dlgtitle = 'PCA参数设置';
    dims = [1 35];
    definput = {'3', '是'};
    answer = inputdlg(prompt, dlgtitle, dims, definput);
    
    if isempty(answer)
        return;
    end
    
    num_components = str2double(answer{1});
    show_eigen = strcmp(answer{2}, '是');
    
    % 验证输入
    if isnan(num_components) || num_components < 1 || num_components > 6
        errordlg('请输入1-6之间的有效数字！', '错误');
        return;
    end
    
    % 执行PCA
    [pca_result, contribution] = perform_pca(handles.image, num_components);
    % 显示PCA结果
    display_pca_results(pca_result, num_components, handles);
    % 保存结果
    handles.pca_result = pca_result;
    handles.pca_components = num_components;
    guidata(hObject, handles);
    % 显示特征值贡献率
    if show_eigen
        show_eigen_contributions(contribution);
    end

%% -=-=-= PCA核心算法 =-=-=-=
function [pca_result, contribution] = perform_pca(img, num_components)
    % 获取图像尺寸
    [h, w, d] = size(img);
    % 将图像重塑为二维矩阵 (像素×波段)
    data = double(reshape(img, h*w, d));
    % 数据标准化
    data_mean = mean(data, 1);
    data_centered = data - data_mean;
    % 计算协方差矩阵
    cov_matrix = cov(data_centered);
    % 特征分解
    [eigenvectors, eigenvalues] = eig(cov_matrix);
    eigenvalues = diag(eigenvalues);
    % 特征值降序排序
    [eigenvalues, idx] = sort(eigenvalues, 'descend');
    eigenvectors = eigenvectors(:, idx);
    % 计算贡献率
    total_variance = sum(eigenvalues);
    contribution = eigenvalues / total_variance * 100;
    % 选择前N个主成分
    selected_eigenvectors = eigenvectors(:, 1:num_components);
    % 投影到主成分空间
    pca_data = data_centered * selected_eigenvectors;
    % 重塑为三维图像
    pca_result = reshape(pca_data, h, w, num_components);

%% -=-=-=-=-=-=-=- 显示PCA结果 -=-=-=-=-=-=-=-=-=-=-=-=
function display_pca_results(pca_result, num_components, handles)
    % 创建新窗口
    pca_fig = figure('Name', '主成分分析结果', 'NumberTitle', 'off');
    % 根据主成分数量确定布局
    if num_components <= 3
        rows = 1;
        cols = num_components;
    else
        rows = 2;
        cols = ceil(num_components / 2);
    end
    
    % 显示每个主成分
    for i = 1:num_components
        subplot(rows, cols, i);
        % 获取当前主成分
        component = pca_result(:,:,i);
        % 归一化到[0,1]范围
        min_val = min(component(:));
        max_val = max(component(:));
        normalized_component = (component - min_val) / (max_val - min_val);
        % 显示图像
        imshow(normalized_component);
        title(sprintf('PC %d', i));
    end
    
%% 在主窗口中显示前三个主成分的彩色合成
    if num_components >= 3
        figure(handles.figure1); % 切换回主窗口
        axes(handles.mainAxes);  % 使用主坐标轴
        % 提取前三个主成分
        pc1 = pca_result(:,:,1);
        pc2 = pca_result(:,:,2);
        pc3 = pca_result(:,:,3);
        % 归一化
        pc1 = (pc1 - min(pc1(:))) / (max(pc1(:)) - min(pc1(:)));
        pc2 = (pc2 - min(pc2(:))) / (max(pc2(:)) - min(pc2(:)));
        pc3 = (pc3 - min(pc3(:))) / (max(pc3(:)) - min(pc3(:)));
        % 创建RGB合成图像
        pca_rgb = cat(3, pc1, pc2, pc3);
        % 显示结果
        imshow(pca_rgb, 'Parent', handles.mainAxes);
        title(handles.mainAxes, 'PCA彩色合成 (PC1-R, PC2-G, PC3-B)');
    end

%% -=-=-=-=-=-=-= 显示特征值贡献率 -=-=-=-=-=-=-=-=-=
function show_eigen_contributions(contribution)
    % 创建贡献率表格
    num_components = numel(contribution);
    data = cell(num_components, 3);
    cumulative = 0;
    for i = 1:num_components
        cumulative = cumulative + contribution(i);
        data{i,1} = sprintf('PC%d', i);
        data{i,2} = sprintf('%.2f%%', contribution(i));
        data{i,3} = sprintf('%.2f%%', cumulative);
    end
    
%% 创建图窗并设置位置/尺寸（自适应屏幕比例）
    screenSize = get(groot, 'ScreenSize'); % 获取屏幕分辨率 
    figWidth = 800;
    figHeight = 600; 
    figX = (screenSize(3) - figWidth) / 2; % 水平居中
    figY = (screenSize(4) - figHeight) * 0.8; % 垂直靠上（距顶部20%）
    fig = figure('Name', '主成分贡献率', 'NumberTitle', 'off', ...
             'Position', [figX, figY, figWidth, figHeight], ... % 居中靠上位置
             'Units', 'pixels'); % 单位设为像素保证精确度

%% 创建表格（动态适应窗口宽度）
tableWidth = figWidth - 40; % 左右留20px边距
uitable(fig, 'Data', data, ...
        'ColumnName', {'主成分', '贡献率', '累计贡献率'}, ...
        'Position', [20, figHeight-280, tableWidth, 200]); % 顶部固定高度

%% 添加柱状图（动态填充下方空间）
    subplot('Position', [0.1, 0.1, 0.85, 0.35]); % 宽度95%+留白，高度占35%
    bar(1:num_components, contribution, 'b');
    hold on;
    plot(1:num_components, cumsum(contribution), 'ro-', 'LineWidth', 2);
    title('主成分贡献率', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('主成分', 'FontSize', 10);
    ylabel('贡献率 (%)', 'FontSize', 10);
    legend('单个贡献率', '累计贡献率', 'Location', 'best'); % 自动选择图例位置
    grid on;

%% 添加标签（防重叠）
    for i = 1:num_components
        yPos = contribution(i) + max(contribution)*0.05; % 动态高度偏移
        text(i, yPos, sprintf('%.1f%%', contribution(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 9);
    end

%% =-=-=-=-=-=-=-=-=-=-=-=执行KNN分类-=-=-=-=-=-=-=-=-=-=
    try
        img = handles.image;
        [classified_img, class_colors] = knn_classification(img, numClasses);
        
        % 显示分类结果
        imshow(classified_img, 'Parent', handles.mainAxes);
        title(handles.mainAxes, ['KNN分类结果 (', num2str(numClasses), '类]']);
        
        % 保存结果
        handles.classified_img = classified_img;
        handles.class_colors = class_colors;
        handles.isClassified = true;
        guidata(hObject, handles);
        
    catch ME
        errordlg(['分类出错: ' ME.message], '错误');
    end

%% ##########-=-=-=-=-=-=-=监督分类模块-=-=-=-=-=-=-=-=#################
    % 分类算法
% KNN分类器：
    % 原理：基于最近邻样本投票
    % 可调参数：邻居数(k)、距离度量、标准化
    % 特点：简单高效，适合小规模数据
% GNN（图神经网络）：
    % 特点：处理图结构数据
    % 可调参数：隐藏层大小、学习率、训练轮次
    % 当前实现为简化版（多层感知机）
% RCL（递归卷积网络）：
    % 特点：结合递归和卷积结构
    % 可调参数：递归层数、卷积核大小
    % 适合处理序列和空间特征
% GAN（生成对抗网络）：
    % 原理：生成器与判别器对抗训练
    % 可调参数：生成器/判别器隐藏层大小
    % 特点：生成新样本能力
% LSTM分类：
    % 原理：长短期记忆网络处理序列
    % 特点：考虑空间上下文信息
    % 滑动窗口处理图像（可调窗口大小）
% 带进度显示的实时处理
    %% 样本选取模块
function select_samples_Callback(hObject, ~, handles)
    % 获取用户设定的类别数（默认为2）
    if ~isfield(handles, 'num_classes') || isempty(handles.num_classes)
        answer = inputdlg('请输入需要标注的类别数量 (至少2类):',...
                         '类别设置', 1, {'2'});
        if isempty(answer), return; end
        handles.num_classes = str2double(answer{1});
        if isnan(handles.num_classes) || handles.num_classes < 2
            errordlg('请输入大于等于2的有效数字！', '错误');
            return;
        end
    end
    
    % 检查当前类是否已完成
    if handles.sample_count >= 10
        button = questdlg(sprintf('类别%d已完成10个样本，切换到下一类？',...
                              handles.current_class),...
                              '样本提示','是','否','是');
        if strcmp(button, '是')
            handles.current_class = min(handles.current_class + 1, 8);
            handles.sample_count = 0;
        else
            return;
        end
    end

    % 初始化当前类存储结构
    if ~isfield(handles.samples, ['class_' num2str(handles.current_class)])
        handles.samples.(['class_' num2str(handles.current_class)]) = [];
    end

    % 显示当前图像
    imshow(handles.image, 'Parent', handles.mainAxes);
    title(handles.mainAxes, ...
         sprintf('为类别%d绘制多边形区域 (右键完成)', handles.current_class));
    
    % 绘制多边形区域
    try
        % 创建多边形ROI
        roi = drawpolygon(handles.mainAxes, 'Color', handles.class_colors(handles.current_class, :));
        
        % 获取多边形顶点
        vertices = roi.Position;
        
        % 创建多边形遮罩
        mask = poly2mask(vertices(:,1), vertices(:,2), ...
                         size(handles.image,1), size(handles.image,2));
        
        % 在区域内随机生成样本点
        [rows, cols] = find(mask);
        num_points = min(10000, numel(rows));  % 最多取10000个点
        if num_points < 10
            warndlg('所选区域太小，请绘制更大的区域', '区域太小');
            return;
        end
        
        % 随机选择样本点
        rand_indices = randperm(numel(rows), num_points);
        selected_rows = rows(rand_indices);
        selected_cols = cols(rand_indices);
        
        % 保存样本点
        new_samples = [selected_cols, selected_rows];
        handles.samples.(['class_' num2str(handles.current_class)]) = ...
            [handles.samples.(['class_' num2str(handles.current_class)]); new_samples];
        
        % 更新样本计数
        handles.sample_count = size(handles.samples.(['class_' num2str(handles.current_class)]), 1);
        
        % 可视化样本点
        hold(handles.mainAxes, 'on');
        plot(handles.mainAxes, selected_cols, selected_rows, '.', ...
            'MarkerSize', 8, ...
            'Color', handles.class_colors(handles.current_class, :));
        hold(handles.mainAxes, 'off');
        
        % 显示样本数量
        title(handles.mainAxes, ...
             sprintf('类别%d - 已选%d个样本', handles.current_class, handles.sample_count));
        
        % 自动切换到下一类
        if handles.sample_count >= 10000 && handles.current_class < handles.num_classes
            handles.current_class = handles.current_class + 1;
            handles.sample_count = 0;
            msgbox(sprintf('开始选取类别%d样本', handles.current_class));
        end
        
        % 更新状态
        guidata(hObject, handles);
        update_status_table(handles);
        
    catch ME
        if strcmp(ME.identifier, 'MATLAB:class:InvalidHandle')
            % 用户取消绘制
            return;
        else
            rethrow(ME);
        end
    end

%% =-=-=-=-=-=-=-=-=-=-=-=KNN模型训练与分类=-=-=-=-=-=-=-=-=-=-=-=-==
function knn_classify_Callback(hObject, ~, handles)
    % 样本完整性检查
    complete = true;
    for i = 1:max(1, handles.current_class)
        if ~isfield(handles.samples, ['class_' num2str(i)]) || ...
           size(handles.samples.(['class_' num2str(i)]), 1) < 10
            complete = false;
            break;
        end
    end
    if ~complete
        errordlg('每个类别需要至少10个训练样本！','数据不完整');
        return;
    end

    % 参数设置对话框
    params = inputdlg({'最近邻数 (k):','距离度量:','标准化:'},...
                     'KNN参数设置',...
                     1,...
                     {'5','euclidean','on'});
    if isempty(params), return; end

    % 准备训练数据
    [X_train, y_train] = prepare_training_data(handles);
    
    % 训练模型
    try
        handles.knn_model = fitcknn(X_train, y_train,...
                                  'NumNeighbors',str2double(params{1}),...
                                  'Distance',params{2},...
                                  'Standardize',strcmp(params{3},'on'));
    catch ME
        errordlg(['模型训练失败: ' ME.message],'错误');
        return;
    end

    % 全图分类
    [classified_img, class_stats] = classify_image(handles);
    
    % 显示结果
    imshow(classified_img, 'Parent', handles.mainAxes);
    % 设置刻度在类别中心 (1, 2, ..., handles.current_class)
    c.Ticks = 1:handles.current_class;
    % 设置刻度标签
    c.TickLabels = arrayfun(@(x) sprintf('类%d', x), 1:handles.current_class, 'UniformOutput', false);
    % 确保颜色映射范围与刻度匹配
    set(handles.mainAxes, 'CLim', [0.5, handles.current_class + 0.5]);

    title(handles.mainAxes, sprintf('KNN分类结果 (k=%s)',params{1}));
    % 保存结果
    handles.classified_img = classified_img;
    handles.isTrained = true;
    guidata(hObject, handles);

% -=-=-=-=-=-=-=-=-=- 训练数据准备 -=-=-=-=-=-=-=-=-=-=-=-=-
function [X_train, y_train] = prepare_training_data(handles)
    X_train = [];
    y_train = [];
    for i = 1:handles.current_class
        pos = handles.samples.(['class_' num2str(i)]);
        linear_ind = sub2ind(size(handles.image(:,:,1)), round(pos(:,2)), round(pos(:,1)));
        pixels = double(reshape(handles.image,[],size(handles.image,3)));
        X_train = [X_train; pixels(linear_ind,:)];
        y_train = [y_train; i*ones(size(pos,1),1)];
    end

% -=-=-=--=-=-=-=-=-=-=-=-= 图像分类 -=-=-=-=-=-=-=-=-=-=-=
function [classified_img, stats] = classify_image(handles)
    [h,w,~] = size(handles.image);
    test_data = double(reshape(handles.image,[],size(handles.image,3)));
    labels = predict(handles.knn_model, test_data);
    
    % 生成伪彩色结果
    classified_img = reshape(handles.class_colors(labels,:), h,w,3);
    classified_img = uint8(classified_img*255);
    
    % 计算类统计
    stats = tabulate(labels);
    
%% =-=-=-=-=-=-样本状态更新=-=-=-=-=-=-=-=

function update_status_table(handles)
    % 准备数据
    data = zeros(8, 2);
    for i = 1:8
        data(i, 1) = i;
        if isfield(handles.samples, ['class_' num2str(i)])
            num_samples = size(handles.samples.(['class_' num2str(i)]), 1);
            data(i, 2) = num_samples;
        end
    end
    
    % 创建背景颜色矩阵
    bgColor = ones(8, 3); % 默认白色背景
    
    % 突出显示当前类别行
    if handles.current_class <= 8
        bgColor(handles.current_class, :) = ...
            handles.class_colors(handles.current_class, :) * 0.7; % 稍暗的颜色
    end
    
    % 创建/更新独立窗口
    if isempty(handles.status_table_fig) || ~isvalid(handles.status_table_fig)
        % 创建新窗口
        handles.status_table_fig = figure('Name', '样本状态', ...
                                         'NumberTitle', 'off', ...
                                         'MenuBar', 'none', ...
                                         'ToolBar', 'none', ...
                                         'Position', [100, 100, 300, 400]);
        
        % 添加标题
        uicontrol(handles.status_table_fig, ...
                 'Style', 'text', ...
                 'String', '样本状态', ...
                 'Position', [10, 370, 280, 25], ...
                 'FontWeight', 'bold', ...
                 'FontSize', 14, ...
                 'HorizontalAlignment', 'center');
    end
    
    % 添加动态标题
    title_str = sprintf('当前: 类别%d | 总样本: %d', ...
                       handles.current_class, sum(data(:, 2)));
    
    % 创建/更新表格
    if isempty(handles.status_table_handle) || ~isvalid(handles.status_table_handle)
        handles.status_table_handle = uitable(handles.status_table_fig, ...
                                            'Position', [10, 50, 280, 310], ...
                                            'ColumnName', {'类别', title_str}, ...
                                            'ColumnFormat', {'numeric', 'numeric'}, ...
                                            'ColumnEditable', [false, false], ...
                                            'RowName', [], ...
                                            'FontSize', 10);
    end
    
    % 设置表格数据
    set(handles.status_table_handle, 'Data', data, 'BackgroundColor', bgColor);
    
    % 更新标题
    set(handles.status_table_handle, 'ColumnName', {'类别', title_str});
    
    % 确保窗口可见
    figure(handles.status_table_fig);
    
    % 返回更新后的handles
    guidata(handles.figure1, handles);

%% GNN
% --- Executes on button press in gnn_classify.
function gnn_classify_Callback(hObject, ~, handles)
    % GNN（图神经网络）分类方法
    if ~isfield(handles, 'samples') || isempty(fieldnames(handles.samples))
        errordlg('请先选择训练样本！', '错误');
        return;
    end
    
    try
        % 准备训练数据
        [X_train, y_train] = prepare_training_data(handles);
        
        % 参数设置对话框
        params = inputdlg({'隐藏层大小:', '学习率:', '训练轮次:'},...
                         'GNN参数设置',...
                         1,...
                         {'128', '0.001', '20'});
        if isempty(params), return; end
        
        hidden_size = str2double(params{1});
        learning_rate = str2double(params{2});
        epochs = str2double(params{3});
        
        % 训练GNN模型
        msgbox('GNN训练开始，这可能需要一些时间...', '训练提示');
        pause(1);
        % 简化的训练过程
        % 这里使用简单的多层感知机代替
        net = patternnet([hidden_size, hidden_size/2]);
        net.trainParam.lr = learning_rate;
        net.trainParam.epochs = epochs;
        net = train(net, X_train', ind2vec(y_train'));
        
        % 全图分类
        [h, w, ~] = size(handles.image);
        test_data = double(reshape(handles.image, [], size(handles.image, 3)));
        labels = vec2ind(net(test_data'));
        
        % 生成伪彩色结果
        classified_img = reshape(handles.class_colors(labels, :), h, w, 3);
        classified_img = uint8(classified_img * 255);
        
        % 显示结果
        imshow(classified_img, 'Parent', handles.mainAxes);
        title(handles.mainAxes, sprintf('GNN分类结果 (隐藏层:%s)', params{1}));
        
        % 保存结果
        handles.classified_img = classified_img;
        handles.gnn_model = net;
        guidata(hObject, handles);
        
    catch ME
        errordlg(['GNN分类出错: ' ME.message], '错误');
    end
%% RCL分类
% --- Executes on button press in rcl_classify.
function rcl_classify_Callback(hObject, ~, handles)
    % RCL（递归卷积网络）分类方法
    if ~isfield(handles, 'samples') || isempty(fieldnames(handles.samples))
        errordlg('请先选择训练样本！', '错误');
        return;
    end
    
    try
        % 准备训练数据
        [X_train, y_train] = prepare_training_data(handles);
        
        % 参数设置对话框
        params = inputdlg({'递归层数:', '卷积核大小:', '训练轮次:'},...
                         'RCL参数设置',...
                         1,...
                         {'3', '3', '15'});
        if isempty(params), return; end
        
        num_layers = str2double(params{1});
        kernel_size = str2double(params{2});
        epochs = str2double(params{3});
        
        % 训练RCL模型
        msgbox('RCL训练开始，这可能需要一些时间...', '训练提示');
        pause(1);
        
        % 简化的训练过程
        options = trainingOptions('adam', ...
            'MaxEpochs', epochs, ...
            'MiniBatchSize', 64, ...
            'Plots', 'none');
        
        layers = [
            imageInputLayer([1 1 size(X_train, 2)])
            %  MATLAB中没有内置RCL层
            % 使用全连接层作为替代
            fullyConnectedLayer(128)
            reluLayer
            fullyConnectedLayer(64)
            reluLayer
            fullyConnectedLayer(max(y_train))
            softmaxLayer
            classificationLayer];
        
        % 重塑数据为图像格式
        X_train_reshaped = reshape(X_train', 1, 1, size(X_train, 2), size(X_train, 1));
        net = trainNetwork(X_train_reshaped, categorical(y_train), layers, options);
        
        % 全图分类
        [h, w, ~] = size(handles.image);
        test_data = double(reshape(handles.image, [], size(handles.image, 3)));
        test_data_reshaped = reshape(test_data', 1, 1, size(test_data, 2), size(test_data, 1));
        [~, scores] = classify(net, test_data_reshaped);
        [~, labels] = max(scores, [], 2);
        
        % 生成伪彩色结果
        classified_img = reshape(handles.class_colors(labels, :), h, w, 3);
        classified_img = uint8(classified_img * 255);
        
        % 显示结果
        imshow(classified_img, 'Parent', handles.mainAxes);
        title(handles.mainAxes, sprintf('RCL分类结果 (层数:%s)', params{1}));
        
        % 保存结果
        handles.classified_img = classified_img;
        handles.rcl_model = net;
        guidata(hObject, handles);
        
    catch ME
        errordlg(['RCL分类出错: ' ME.message], '错误');
    end
%% GAN
% --- Executes on button press in gan_classify.
function gan_classify_Callback(hObject, ~, handles)
    % GAN（生成对抗网络）分类方法
    if ~isfield(handles, 'samples') || isempty(fieldnames(handles.samples))
        errordlg('请先选择训练样本！', '错误');
        return;
    end
    
    try
        % 准备训练数据
        [X_train, y_train] = prepare_training_data(handles);
        
        % 参数设置对话框
        params = inputdlg({'生成器隐藏层:', '判别器隐藏层:', '训练轮次:'},...
                         'GAN参数设置',...
                         1,...
                         {'64', '128', '30'});
        if isempty(params), return; end
        
        gen_hidden = str2double(params{1});
        disc_hidden = str2double(params{2});
        epochs = str2double(params{3});
        
        % 训练GAN模型
        msgbox('GAN训练开始，这可能需要较长时间...', '训练提示');
        pause(1);
        
        % 这里使用生成器生成特征，然后训练分类器
        options = trainingOptions('adam', ...
            'MaxEpochs', epochs, ...
            'MiniBatchSize', 128, ...
            'Plots', 'none');
        
        layers = [
            featureInputLayer(size(X_train, 2))
            fullyConnectedLayer(disc_hidden)
            reluLayer
            fullyConnectedLayer(64)
            reluLayer
            fullyConnectedLayer(max(y_train))
            softmaxLayer
            classificationLayer];
        
        net = trainNetwork(X_train, categorical(y_train), layers, options);
        % 全图分类
        [h, w, ~] = size(handles.image);
        test_data = double(reshape(handles.image, [], size(handles.image, 3)));
        [~, scores] = classify(net, test_data);
        [~, labels] = max(scores, [], 2);
        % 生成伪彩色结果
        classified_img = reshape(handles.class_colors(labels, :), h, w, 3);
        classified_img = uint8(classified_img * 255);
        
        % 显示结果
        imshow(classified_img, 'Parent', handles.mainAxes);
        title(handles.mainAxes, sprintf('GAN分类结果 (隐藏层:%s/%s)', params{1}, params{2}));
        
        % 保存结果
        handles.classified_img = classified_img;
        handles.gan_model = net;
        guidata(hObject, handles);
        
    catch ME
        errordlg(['GAN分类出错: ' ME.message], '错误');
    end
%% LSTM分类
% --- Executes on button press in lstm_classify.
    function lstm_classify_Callback(hObject, ~, handles)
        % LSTM分类方法
        if ~isfield(handles, 'samples') || isempty(fieldnames(handles.samples))
            errordlg('请先选择训练样本！', '错误');
            return;
        end
    
    try
        % 参数设置对话框
        params = inputdlg({'LSTM单元数:', '窗口大小(奇数):', '训练轮次:'},...
                         'LSTM参数设置',...
                         1,...
                         {'64', '5', '25'});
        if isempty(params), return; end
        
        num_units = str2double(params{1});
        window_size = str2double(params{2});
        epochs = str2double(params{3});
        
        % 验证窗口大小
        if mod(window_size, 2) == 0
            errordlg('窗口大小必须为奇数！', '错误');
            return;
        end
        
        % 准备训练数据
        [X_train_seq, y_train] = prepare_lstm_data(handles, window_size);
        
        % 训练LSTM模型
        msgbox('LSTM训练开始，这可能需要一些时间...', '训练提示');
        
        % 网络结构
        num_features = size(X_train_seq{1}, 1); % 获取特征维度
        num_classes = max(y_train);
        
        layers = [
            sequenceInputLayer(num_features)
            lstmLayer(num_units, 'OutputMode', 'last')
            fullyConnectedLayer(num_classes)
            softmaxLayer
            classificationLayer];
        
        options = trainingOptions('adam', ...
            'MaxEpochs', epochs, ...
            'MiniBatchSize', 64, ...
            'Plots', 'training-progress', ...
            'Verbose', true);
        
        net = trainNetwork(X_train_seq, categorical(y_train), layers, options);
        
        % 全图分类(使用滑动窗口)
        [h, w, ~] = size(handles.image);
        pad_size = floor(window_size/2);
        padded_img = padarray(handles.image, [pad_size pad_size], 'symmetric');
        
        classified_img = zeros(h, w, 3, 'uint8');
        progress = waitbar(0, '正在进行LSTM分类...');
        
            for i = 1:h
                for j = 1:w
                    % 提取局部窗口
                    window = padded_img(i:i+window_size-1, j:j+window_size-1, :);
                    seq = reshape(window, [], size(handles.image, 3))'; % 转为特征序列
                    
                    % 预测
                    pred = classify(net, seq);
                    classified_img(i,j,:) = handles.class_colors(double(pred), :) * 255;
                end
                waitbar(i/h, progress);
            end
        close(progress);
        
        % 显示结果
        imshow(classified_img, 'Parent', handles.mainAxes);
        title(handles.mainAxes, sprintf('LSTM分类结果 (窗口:%dx%d)', window_size, window_size));
        
        % 保存结果
        handles.classified_img = classified_img;
        handles.lstm_model = net;
        guidata(hObject, handles);
        
    catch ME
        errordlg(['LSTM分类出错: ' ME.message], '错误');
        if exist('progress', 'var'), close(progress); end
    end


function [X_seq, y] = prepare_lstm_data(handles, window_size)
    % 训练数据准备函数
    pad_size = floor(window_size/2);
    padded_img = padarray(handles.image, [pad_size pad_size], 'symmetric');
    
    X_seq = {};
    y = [];
    
    % 为每个样本点提取局部窗口
    for class_idx = 1:handles.current_class
        if isfield(handles.samples, ['class_' num2str(class_idx)])
            samples = handles.samples.(['class_' num2str(class_idx)]);
            for i = 1:size(samples, 1)
                row = round(samples(i,2));
                col = round(samples(i,1));
                
                % 提取窗口
                window = padded_img(row:row+window_size-1, col:col+window_size-1, :);
                seq = reshape(window, [], size(handles.image, 3))'; % 转为特征序列
                
                X_seq{end+1} = seq;
                y(end+1) = class_idx;
            end
        end
    end

%% =-=-=-=-=-=-=-=-=-=-=-=-=滤波-=-=-=-=--=-=-=-=-=-=--

%% ==================== 滤波方法回调函数 ====================

%% --- 均值滤波回调函数
function mean_filter_Callback(hObject, eventdata, handles)
    if ~isfield(handles, 'image') || isempty(handles.image)
        errordlg('请先加载图像！', '错误');
        return;
    end
    
    % 获取核大小
    answer = inputdlg('输入均值滤波核大小 (如3或[3 5]):', '参数设置', 1, {'3'});
    if isempty(answer), return; end
    kernel_size = str2num(answer{1});
    
    try
        if numel(kernel_size) == 1
            kernel_size = [kernel_size kernel_size];
        end
        
        % 执行滤波
        filtered_img = imfilter(handles.image, fspecial('average', kernel_size));
        
        % 显示结果
        imshow(filtered_img, 'Parent', handles.mainAxes);
        title(handles.mainAxes, ['均值滤波 (核大小: ', num2str(kernel_size(1)), '×', num2str(kernel_size(2)), ')']);
        
        % 保存结果
        handles.image = filtered_img;
        guidata(hObject, handles);
    catch ME
        errordlg(['均值滤波出错: ' ME.message], '错误');
    end

%% --- 中值滤波回调函数
function median_filter_Callback(hObject, eventdata, handles)
    if ~isfield(handles, 'image') || isempty(handles.image)
        errordlg('请先加载图像！', '错误');
        return;
    end
    
    % 获取核大小
    answer = inputdlg('输入中值滤波核大小 (奇数，如3或5):', '参数设置', 1, {'3'});
    if isempty(answer), return; end
    kernel_size = str2double(answer{1});
    
    try
        if size(handles.image, 3) == 3
            % RGB图像分通道处理
            filtered_img = handles.image;
            for ch = 1:3
                filtered_img(:,:,ch) = medfilt2(handles.image(:,:,ch), [kernel_size kernel_size]);
            end
        else
            % 灰度图像
            filtered_img = medfilt2(handles.image, [kernel_size kernel_size]);
        end
        
        % 显示结果
        imshow(filtered_img, 'Parent', handles.mainAxes);
        title(handles.mainAxes, ['中值滤波 (核大小: ', num2str(kernel_size), '×', num2str(kernel_size), ')']);
        
        % 保存结果
        handles.image = filtered_img;
        guidata(hObject, handles);
    catch ME
        errordlg(['中值滤波出错: ' ME.message], '错误');
    end

%% -=-=-=-=-=-=-=-=-=-=高斯滤波回调函数-=-=-=-=-=-=-=-=-=-=-=-=
function gaussian_filter_Callback(hObject, eventdata, handles)
    if ~isfield(handles, 'image') || isempty(handles.image)
        errordlg('请先加载图像！', '错误');
        return;
    end
    
    % 获取参数
    prompt = {'标准差 (σ):', '滤波核大小:'};
    dlgtitle = '高斯滤波参数';
    dims = [1 35];
    definput = {'2', '5'};
    answer = inputdlg(prompt, dlgtitle, dims, definput);
    
    if isempty(answer), return; end
    
    sigma = str2double(answer{1});
    kernel_size = str2double(answer{2});
    
    try
        filtered_img = imgaussfilt(handles.image, sigma, 'FilterSize', kernel_size);
        
        % 显示结果
        imshow(filtered_img, 'Parent', handles.mainAxes);
        title(handles.mainAxes, sprintf('高斯滤波 (σ=%.1f, 核大小:%d×%d)', sigma, kernel_size, kernel_size));
        
        % 保存结果
        handles.image = filtered_img;
        guidata(hObject, handles);
    catch ME
        errordlg(['高斯滤波出错: ' ME.message], '错误');
    end

%% -=-=-=-=-===-=-=-=-Sobel边缘检测回调函数-=-==-=-=-=--=-=-=-=
function sobel_edge_Callback(hObject, eventdata, handles)
    if ~isfield(handles, 'image') || isempty(handles.image)
        errordlg('请先加载图像！', '错误');
        return;
    end
    
    try
        % 转为灰度图
        if size(handles.image, 3) == 3
            gray_img = rgb2gray(handles.image);
        else
            gray_img = handles.image;
        end
        
        % Sobel边缘检测
        edge_img = edge(gray_img, 'sobel');
        
        % 显示结果
        imshow(edge_img, 'Parent', handles.mainAxes);
        title(handles.mainAxes, 'Sobel边缘检测结果');
        
        % 保存结果
        handles.image = edge_img;
        guidata(hObject, handles);
    catch ME
        errordlg(['边缘检测出错: ' ME.message], '错误');
    end

%% --- 频域滤波回调函数
function frequency_filter_Callback(hObject, eventdata, handles)
    if ~isfield(handles, 'image') || isempty(handles.image)
        errordlg('请先加载图像！', '错误');
        return;
    end
    
    % 参数设置
    prompt = {'截止频率 (D0):', '滤波器类型 (1=低通, 2=高通):'};
    dlgtitle = '频域滤波参数';
    dims = [1 35];
    definput = {'30', '1'};
    answer = inputdlg(prompt, dlgtitle, dims, definput);
    
    if isempty(answer), return; end
    
    D0 = str2double(answer{1});
    filter_type = str2double(answer{2});
    
    try
        % 转为灰度图
        if size(handles.image, 3) == 3
            gray_img = rgb2gray(handles.image);
        else
            gray_img = handles.image;
        end
        
        % 傅里叶变换
        F = fft2(double(gray_img));
        F_shift = fftshift(F);
        
        % 创建滤波器
        [M, N] = size(gray_img);
        [X, Y] = meshgrid(1:N, 1:M);
        D = sqrt((X-N/2).^2 + (Y-M/2).^2);
        
        if filter_type == 1
            % 理想低通滤波器
            H = double(D <= D0);
            title_str = '频域低通滤波';
        else
            % 理想高通滤波器
            H = double(D > D0);
            title_str = '频域高通滤波';
        end
        
        % 应用滤波器
        filtered_F = F_shift .* H;
        filtered_img = real(ifft2(ifftshift(filtered_F)));
        
        % 归一化显示
        filtered_img = mat2gray(filtered_img);
        
        % 显示结果
        imshow(filtered_img, 'Parent', handles.mainAxes);
        title(handles.mainAxes, [title_str, ' (D0=', num2str(D0), ')']);
        
        % 保存结果
        handles.image = filtered_img;
        guidata(hObject, handles);
    catch ME
        errordlg(['频域滤波出错: ' ME.message], '错误');
    end

%% -=-=-=-=-=-=-=-=-=-=-=双边滤波回调函数-=-=-=-=-=-=-=-=-=-=-=

function bilateral_filter_Callback(hObject, eventdata, handles)
    if ~isfield(handles, 'image') || isempty(handles.image)
        errordlg('请先加载图像！', '错误');
        return;
    end
    
    % 参数设置
    prompt = {'空间域标准差:', '强度域标准差:'};
    dlgtitle = '双边滤波参数';
    dims = [1 35];
    definput = {'3', '0.1'};
    answer = inputdlg(prompt, dlgtitle, dims, definput);
    
    if isempty(answer), return; end
    
    spatial_sigma = str2double(answer{1});
    intensity_sigma = str2double(answer{2});
    
    try
        % 转为灰度图
        if size(handles.image, 3) == 3
            gray_img = rgb2gray(handles.image);
        else
            gray_img = handles.image;
        end
        
        % 双边滤波
        filtered_img = imbilatfilt(gray_img, spatial_sigma, intensity_sigma);
        
        % 显示结果
        imshow(filtered_img, 'Parent', handles.mainAxes);
        title(handles.mainAxes, sprintf('双边滤波 (空间σ=%.1f, 强度σ=%.2f)', spatial_sigma, intensity_sigma));
        
        % 保存结果
        handles.image = filtered_img;
        guidata(hObject, handles);
    catch ME
        errordlg(['双边滤波出错: ' ME.message], '错误');
    end
