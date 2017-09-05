%this file denoises mnist images using the OMP algorithm with different number of nonzeros K,
%and saves the denoised images in the same directory
%before running the script, install the ompbox module from here: http://www.cs.technion.ac.il/~ronrubin/software.html

%add the path to the installed library (might be different based on your installation):
addpath '/usr/local/MATLAB/R2017a/toolbox/ompbox10'


omp_K=4 %number of nonzeros in each sparse representation
patch_size=[7,7];
adv_dir='./'%directory to adversarial samples
method='Deepfool'%adversarial algorithm
niters=[1,5];%No. iterations
epsilons=[0.001];%epsilon used
path_to_dict='mnist_dicts.mat'%path to dictionaries




load(path_to_dict);
for i=1:size(class_dictionaries,1)
    dic=class_dictionaries(i,:,:);
    dic=reshape(dic,size(class_dictionaries,2),size(class_dictionaries,3));
    class_dictionaries(i,:,:)=normc(dic);
end

num_classes=size(class_dictionaries,1)
for iter=niters
    for eps_id=1:length(epsilons)
        eps=epsilons(eps_id)
        images_directory=[adv_dir,method,'/',num2str(iter),'iterations/eps',num2str(eps),'/perturbed.mat'];
        %images_directory='original_data/mnist.mat'

        if exist(images_directory)~=2
            continue
        end
        load(images_directory)

        K=omp_K
        denoised_images=zeros(size(perturbed_images));
        for i=1:num_classes
            inds=find(missed_to==i-1);
            if length(inds)==0
                continue;
            end
            images=perturbed_images(inds,:);
            images=reshape(images,size(images,1),28,28);
            images=permute(images,[3,2,1]);
            dic=class_dictionaries(i,:,:);
            dic=reshape(dic,size(dic,2),size(dic,3));
            denoised=image_denoise(images,patch_size,dic,K);
            denoised=permute(denoised,[3,2,1]);
            denoised=reshape(denoised,size(denoised,1),784);
            denoised_images(inds,:)=denoised;
        end
        images_directory=[method,'/',num2str(iter),'iterations/eps',num2str(eps),'/denoised',num2str(K),'.mat'];
        %images_directory=['original_data','/denoised',num2str(K),'.mat']

        parsave(images_directory,denoised_images,original_images,missed_from,missed_to)

    end
end
