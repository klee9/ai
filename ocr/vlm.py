import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, Gemma3ForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

"""
SigLIP2 + Gemma 3 1B 모델을 결합한 VLM 모델입니다.

Gemma 3 270M, 1B는 경량화 모델이지만, 기본적으로 멀티모달 구조가 아니기 때문에 이미지를 입력으로 받을 수 없습니다.
따라서 비전 인코더를 추가하여 이미지를 임베딩 토큰으로 변환한 후, 프로젝터를 통해 Gemma 3의 언어 모델이 처리할 수 있는 차원으로 매칭해야 합니다.
이를 위해 SigLIP2의 비전 인코더를 사용하였으며, Gemma 3의 언어 모델과 차원 수를 맞추기 위해 선형 프로젝터를 추가하였습니다.
"""

class VLM(nn.Module):
    def __init__(self, vision_model_id, language_model_id):
        super().__init__()
        
        # vision encoder
        self.vision_encoder = AutoModel.from_pretrained(vision_model_id, local_files_only=True)
        self.vision_processor = AutoProcessor.from_pretrained(vision_model_id, local_files_only=True)
        # self.vision_processor.image_processor.size = {"shortest_edge": 384} 
        # self.vision_processor.image_processor.max_image_size = {"longest_edge": 1024}
        
        # language model
        self.language_model = Gemma3ForCausalLM.from_pretrained(language_model_id, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_id, local_files_only=True)
        
        # projector
        vision_dim = self.vision_encoder.config.vision_config.hidden_size
        text_dim = self.language_model.config.hidden_size
        
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
        ).to(dtype=self.language_model.dtype)

        # freeze the backbones
        self.vision_encoder.requires_grad_(False)
        self.language_model.requires_grad_(False)

        # only train the projector
        for p in self.projector.parameters():
            p.requires_grad = True

    def forward(self, pixel_values, input_ids, attention_mask=None, spatial_shape=None, labels=None):
        """
          Forward pass method for training

          Args: 
            - pixel_values: preprocessed image tensor
            - input_ids: tokenized text input ids
            - attention_mask: attention mask for text inputs
            - spatial_shape: spatial shape info for vision encoder
            - labels: target labels for language modeling

          Returns:
            - language model outputs
        """
        # extract visual features (previously discarded all spatial info)
        with torch.no_grad():
            vision_outputs = self.vision_encoder.vision_model(
                pixel_values=pixel_values, 
                attention_mask=attention_mask, 
                spatial_shape=spatial_shape
            )
            image_features = vision_outputs.last_hidden_state
        
        # project to text embedding space
        image_embeddings = self.projector(image_features) # [B, Patches, TextDim]
        
        # get text embeddings
        text_embeddings = self.language_model.model.embed_tokens(input_ids) # [B, SeqLen, TextDim]
        
        # concat image and text embeddings
        inputs_embeds = torch.cat([image_embeddings, text_embeddings], dim=1)
        
        # create attention mask for combined embeddings
        batch_size = image_embeddings.shape[0]
        num_patches = image_embeddings.shape[1]
        
        image_mask = torch.ones((batch_size, num_patches), device=attention_mask.device, dtype=attention_mask.dtype)
        combined_mask = torch.cat([image_mask, attention_mask], dim=1)
        
        # handle labels (for training)
        if labels is not None:
            # mask image part in labels with -100 to ignore them
            image_labels = torch.full((batch_size, num_patches), -100, device=labels.device, dtype=labels.dtype)
            combined_labels = torch.cat([image_labels, labels], dim=1)
        else:
            combined_labels = None

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=combined_labels
        )

    @torch.no_grad()
    def generate(self, image: Image.Image, prompt: str, max_new_tokens=64):
        """
          Response generation method

          Args:
            - image: PIL Image input
            - prompt: text prompt
            - max_new_tokens: maximum number of tokens to generate

          Returns:
            - generated text response
        """
        print("Generating response...")
        # preprocess
        inputs = self.vision_processor(images=image, return_tensors="pt").to(self.language_model.device)
        pixel_values = inputs.pixel_values
        spatial_shape = inputs.spatial_shapes
        pixel_attention_mask = inputs.pixel_attention_mask
        
        text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.language_model.device)
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        
        # get visual embeddings
        vision_outputs = self.vision_encoder.vision_model(
            pixel_values=pixel_values, 
            attention_mask=pixel_attention_mask, 
            spatial_shapes=spatial_shape
        )
        image_features = vision_outputs.last_hidden_state.to(dtype=self.language_model.dtype)
        image_features = image_features[:, :64, :]
        image_embeddings = self.projector(image_features)
        
        # get text embeddings
        text_embeddings = self.language_model.model.embed_tokens(input_ids)
        
        # concat embeddings
        inputs_embeds = torch.cat([image_embeddings, text_embeddings], dim=1)
        
        # mask
        image_mask = torch.ones((1, image_embeddings.shape[1]), device=self.language_model.device)
        combined_mask = torch.cat([image_mask, attention_mask], dim=1)
        
        # generate response
        generated_ids = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)