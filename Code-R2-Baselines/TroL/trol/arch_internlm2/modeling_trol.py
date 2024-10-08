# System
import torch
from torch import nn
from utils.utils import *
import torch.utils.checkpoint
from typing import List, Optional, Tuple, Union
from .build_module import build_vision_projector, build_vision_tower
from .modeling_internlm2 import InternLM2Model, InternLM2PreTrainedModel

# Dataclass & ModelOutput
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
@dataclass
class TroLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class TroLForCausalLM(InternLM2PreTrainedModel):
    _auto_class = 'AutoModelForCausalLM'

    _tied_weights_keys = ['output.weight']

    def __init__(self, config):
        super().__init__(config)
        
        # Model
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.max_length = config.max_length

        # Initialize weights and apply final processing
        self.post_init()

        # Vision Encoder
        self.vit = build_vision_tower()

        # Vision Projection
        self.vision_proj = build_vision_projector(self.config.hidden_size)
    
        # image processing variable
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,-1,1,1) * 255
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,-1,1,1) * 255

        # prompt rule
        self.prompt_rule = {"system_start": "<s>[UNUSED_TOKEN_146]system\n",
                            "system_end": "[UNUSED_TOKEN_145]",
                            "user_start": "[UNUSED_TOKEN_146]user\n",
                            "user_end": "[UNUSED_TOKEN_145]",
                            "assistant_start": "[UNUSED_TOKEN_146]assistant\n",
                            "assistant_end": "[UNUSED_TOKEN_145]\n</s>",
                            "test_start": "assistant\n",
                            "test_end": "[UNUSED_TOKEN_145]",
                            "split": "\n",
                            }

    def image_processor(self, images):
        norm_images = (images - self.mean.to(images.device)) / self.std.to(images.device)
        return norm_images

    def eval_process(
        self,
        inputs,
        data,
        tokenizer,
        device,
        img_token_number,
    ):
        batched_image = []
        batched_qa_prompt=[]
        for _input in inputs:

            # Visualization
            # imim = _input['image'].cpu().permute(1, 2, 0)

            # adding <image> to question if not included despite being an image, and adding system prompt and <tor> prompt 
            if 'image' in _input.keys() and not '<image>' in _input['question']: _input['question'] = '<image>\n' + _input['question']

            # make question and answer
            question = make_instruction(_input['question'], data, self.prompt_rule)

            # add bundle image tokens if it has <image> token
            question = add_bundle_tokens(question, '<image>', img_token_number) 

            batched_qa_prompt.append(question)

            # making batched image prompt
            if 'image' in _input.keys() and _input['image'] != None: batched_image.append(_input['image'].to(device))

        '''For Final Outputs'''
        qa_prompts = tokenizer(batched_qa_prompt, padding='longest', return_tensors="pt", add_special_tokens=False)

        # [1] input_ids
        input_ids = qa_prompts.input_ids.to(device)
  
        # [2] attention_mask
        attention_mask = qa_prompts.attention_mask.to(device)

        # [3] im_mask
        im_mask = torch.zeros_like(input_ids).bool()
        im_mask[torch.where(input_ids==self.config.image_token_index)] = True

        if len(batched_image):
            return {"input_ids": input_ids, 
                    "attention_mask": attention_mask, 
                    "im_mask": im_mask,
                    "image_features": self.clip_features(self.image_processor(torch.stack(batched_image)).to(device))
                    }
        else:
            return {"input_ids": input_ids, 
                    "attention_mask": attention_mask, 
                    "im_mask": im_mask,
                    }

    def clip_features(self, image):
        self.vit.eval()
        return self.vit(image)

    def _merge_input_embeds_with_image_features(self, image_features, inputs_embeds, input_ids):
        
        # batch index for image feature
        batch_ind_image_feature = 0

        # shape of image_features
        _, C, D = image_features.shape
        
        # print(image_features.shape)
        # print(inputs_embeds.shape)
        # print(len(input_ids))

        for ind, input_id in enumerate(input_ids):
            matching = torch.where(input_id==self.config.image_token_index)
            #print(matching)
            num_image_tokens_per_one_sample = len(matching[0]) // C
            # print(f"inputs_embeds[{ind}] shape:", inputs_embeds[ind].shape)
            # print(f"Shape of matching indices:", matching[0].shape)
            # print(f"Number of image tokens per sample:", num_image_tokens_per_one_sample)
            # print(f"Image features slice shape:",
            #     image_features[batch_ind_image_feature: batch_ind_image_feature + num_image_tokens_per_one_sample].shape)
            # image_token_embeds = image_features[
            #     batch_ind_image_feature: batch_ind_image_feature + num_image_tokens_per_one_sample
            # ].view(-1, D)
            # print(f"Shape mismatch: {inputs_embeds[ind][matching].shape} vs {image_token_embeds.shape}")
            inputs_embeds[ind][matching] = image_features[batch_ind_image_feature: batch_ind_image_feature+num_image_tokens_per_one_sample].view(-1, D)
            batch_ind_image_feature += num_image_tokens_per_one_sample
    
    
    # def _merge_input_embeds_with_image_features(self, image_features, inputs_embeds, input_ids):
    
    #     # batch index for image feature
    #     batch_ind_image_feature = 0

    #     # shape of image_features
    #     _, C, D = image_features.shape

    #     for ind, input_id in enumerate(input_ids):
    #         # Get the indices where input_id matches the image token index
    #         matching = torch.where(input_id == self.config.image_token_index)
            
    #         # Adjust the number of image tokens to match the number of embeddings being replaced
    #         num_image_tokens_per_one_sample = len(matching) // C # Total matching indices
            
    #         # Debugging shapes
    #         print(f"inputs_embeds[{ind}] shape:", inputs_embeds[ind].shape)
    #         print(f"Shape of matching indices:", matching[0].shape)
    #         print(f"Number of image tokens per sample:", num_image_tokens_per_one_sample)
    #         print(f"Image features slice shape:",
    #             image_features[batch_ind_image_feature: batch_ind_image_feature + num_image_tokens_per_one_sample].shape)
            
    #         # Ensure the shapes match before assignment
    #         image_token_embeds = image_features[
    #             batch_ind_image_feature: batch_ind_image_feature + num_image_tokens_per_one_sample
    #         ].view(-1, D)

    #         if inputs_embeds[ind][matching].shape != image_token_embeds.shape:
    #             print(f"Shape mismatch: {inputs_embeds[ind][matching].shape} vs {image_token_embeds.shape}")
    #         else:
    #             inputs_embeds[ind][matching] = image_token_embeds
            
    #         # Update the batch index for the next sample
    #         batch_ind_image_feature += num_image_tokens_per_one_sample


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        im_mask: torch.BoolTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TroLCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if image_features is not None and input_ids.shape[1] != 1:
                image_features = self.vision_proj(image_features.to(inputs_embeds.dtype))
                self._merge_input_embeds_with_image_features(image_features, inputs_embeds, input_ids)

            # In case input_ids.shape[1] == 1 & image_features==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and image_features is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
                im_mask = torch.zeros(inputs_embeds.shape[:2]).bool().to(inputs_embeds.device)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            im_mask=im_mask,
        )

        hidden_states = outputs[0]
        logits = self.output(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TroLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      attention_mask=None,
                                      inputs_embeds=None,
                                      image_features=None,
                                      im_mask=None,
                                      **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "image_features": image_features,
                "im_mask": im_mask,
            }
        )
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past), )
        return reordered_past