#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VQA数据集生成完整流程
接收batch_process.sh的输出，依次生成问题和答案，生成完整的VQA数据集
"""
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import shutil

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_vqa.generate_question.vqa_generator import VQAGenerator
from generate_vqa.generate_answer.answer_generator import AnswerGenerator
from generate_vqa.generate_answer.validator import AnswerValidator


class VQAPipeline:
    """VQA数据集生成完整流程"""
    
    def __init__(
        self,
        question_config_path: Optional[Path] = None,
        answer_config_path: Optional[Path] = None
    ):
        """
        初始化流程
        
        Args:
            question_config_path: 问题生成配置文件路径
            answer_config_path: 答案生成配置文件路径
        """
        project_root = Path(__file__).parent.parent
        
        # 默认配置文件路径
        if question_config_path is None:
            question_config_path = project_root / "generate_vqa" / "question_config.json"
        if answer_config_path is None:
            answer_config_path = project_root / "generate_vqa" / "generate_answer" / "answer_config.json"
        
        # 初始化生成器
        self.question_generator = VQAGenerator(config_path=question_config_path)
        self.answer_generator = AnswerGenerator(
            config_path=answer_config_path if answer_config_path.exists() else None
        )
        self.validator = AnswerValidator()
        
        # 统计信息
        self.stats = {
            "input_records": 0,
            "questions_generated": 0,
            "answers_generated": 0,
            "validation_passed": 0,
            "validation_failed": 0,
            "errors": []
        }
    
    def _generate_answers(
        self,
        questions_file: Path,
        answers_file: Path
    ) -> None:
        """
        生成答案（内部方法，复用main.py的逻辑）
        """
        import json
        from datetime import datetime
        
        print(f"[INFO] 读取问题文件: {questions_file}")
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        if not isinstance(questions_data, list):
            raise ValueError(f"问题文件应该包含一个数组，但得到: {type(questions_data)}")
        
        print(f"[INFO] 总问题数: {len(questions_data)}")
        
        # 处理每个问题
        results = []
        errors = []
        total_processed = 0
        total_success = 0
        total_failed = 0
        
        for idx, record in enumerate(questions_data, 1):
            total_processed += 1
            
            try:
                # 提取必要信息
                question = record.get("question")
                question_type = record.get("question_type")
                image_base64 = record.get("image_base64")
                
                if not question:
                    errors.append({
                        "index": idx,
                        "id": record.get("id"),
                        "error": "缺少question字段"
                    })
                    total_failed += 1
                    continue
                
                if not question_type:
                    errors.append({
                        "index": idx,
                        "id": record.get("id"),
                        "error": "缺少question_type字段"
                    })
                    total_failed += 1
                    continue
                
                if not image_base64:
                    errors.append({
                        "index": idx,
                        "id": record.get("id"),
                        "error": "缺少image_base64字段"
                    })
                    total_failed += 1
                    continue
                
                # 生成答案
                pipeline_info = {
                    "pipeline_name": record.get("pipeline_name"),
                    "pipeline_intent": record.get("pipeline_intent"),
                    "answer_type": record.get("answer_type")
                }
                
                answer_result = self.answer_generator.generate_answer(
                    question=question,
                    image_base64=image_base64,
                    question_type=question_type,
                    pipeline_info=pipeline_info
                )
                
                if not answer_result or answer_result.get("answer") is None:
                    errors.append({
                        "index": idx,
                        "id": record.get("id"),
                        "error": "答案生成失败"
                    })
                    total_failed += 1
                    continue
                
                # 校验和修复
                validated_result, validation_report = self.validator.validate_and_fix(
                    result=answer_result,
                    image_base64=image_base64
                )
                
                # 构建输出结果
                result = {
                    # 原始信息
                    "question": question,
                    "question_type": question_type,
                    "image_base64": image_base64,
                    
                    # 答案信息（使用校验后的结果）
                    "answer": validated_result.get("answer", answer_result.get("answer")),
                    "explanation": validated_result.get("explanation", answer_result.get("explanation", "")),
                    
                    # 完整问题（选择题包含选项）
                    "full_question": validated_result.get("full_question", answer_result.get("full_question", question)),
                    
                    # 选择题特有字段
                    "options": validated_result.get("options", answer_result.get("options")),
                    "correct_option": validated_result.get("correct_option", answer_result.get("correct_option")),
                    
                    # 校验信息
                    "validation_report": validation_report,
                    
                    # 原始元数据
                    "pipeline_name": record.get("pipeline_name"),
                    "pipeline_intent": record.get("pipeline_intent"),
                    "answer_type": record.get("answer_type"),
                    "sample_index": record.get("sample_index"),
                    "id": record.get("id"),
                    "source_a_id": record.get("source_a_id"),
                    "timestamp": record.get("timestamp", ""),
                    "generated_at": datetime.now().isoformat()
                }
                
                results.append(result)
                total_success += 1
                
                # 进度报告
                if total_processed % 10 == 0:
                    print(f"[进度] 已处理: {total_processed}, 成功: {total_success}, 失败: {total_failed}")
                    
            except Exception as e:
                errors.append({
                    "index": idx,
                    "id": record.get("id"),
                    "error": str(e)
                })
                total_failed += 1
                print(f"[ERROR] 处理记录 {idx} 时出错: {e}")
        
        # 保存结果
        answers_file.parent.mkdir(parents=True, exist_ok=True)
        with open(answers_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存错误信息
        if errors:
            error_file = answers_file.parent / f"{answers_file.stem}_errors.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)
            print(f"  错误信息已保存到: {error_file}")
        
        print(f"[完成] 答案生成完成！")
        print(f"  总处理: {total_processed}")
        print(f"  成功生成: {total_success}")
        print(f"  失败: {total_failed}")
    
    def run(
        self,
        input_file: Path,
        output_dir: Path,
        pipeline_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        save_intermediate: bool = True
    ) -> Dict[str, Any]:
        """
        运行完整流程
        
        Args:
            input_file: 输入文件路径（batch_process.sh的输出）
            output_dir: 输出目录
            pipeline_names: 要使用的pipeline列表（None表示使用所有）
            max_samples: 最大处理样本数（None表示全部）
            save_intermediate: 是否保存中间结果
            
        Returns:
            统计信息和结果路径
        """
        print("=" * 80)
        print("VQA数据集生成完整流程")
        print("=" * 80)
        print(f"输入文件: {input_file}")
        print(f"输出目录: {output_dir}")
        print()
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建中间结果目录
        if save_intermediate:
            intermediate_dir = output_dir / "intermediate"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            questions_dir = intermediate_dir / "questions"
            answers_dir = intermediate_dir / "answers"
            questions_dir.mkdir(exist_ok=True)
            answers_dir.mkdir(exist_ok=True)
        else:
            intermediate_dir = None
            questions_dir = None
            answers_dir = None
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: 生成问题
        print("\n" + "=" * 80)
        print("Step 1: 生成问题")
        print("=" * 80)
        
        questions_file = None
        if save_intermediate:
            questions_file = questions_dir / f"questions_{timestamp}.json"
        else:
            questions_file = output_dir / f"questions_{timestamp}.json"
        
        try:
            self.question_generator.process_data_file(
                input_file=input_file,
                output_file=questions_file,
                pipeline_names=pipeline_names,
                max_samples=max_samples
            )
            
            # 读取生成的问题统计
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
                self.stats["questions_generated"] = len(questions_data)
                self.stats["input_records"] = max_samples or len(json.load(open(input_file, 'r', encoding='utf-8')))
            
            print(f"✓ 问题生成完成: {len(questions_data)} 个问题")
            
        except Exception as e:
            print(f"✗ 问题生成失败: {e}")
            self.stats["errors"].append(f"问题生成失败: {str(e)}")
            raise
        
        # Step 2: 生成答案
        print("\n" + "=" * 80)
        print("Step 2: 生成答案")
        print("=" * 80)
        
        answers_file = None
        if save_intermediate:
            answers_file = answers_dir / f"answers_{timestamp}.json"
        else:
            answers_file = output_dir / f"answers_{timestamp}.json"
        
        try:
            # 直接调用答案生成逻辑
            self._generate_answers(
                questions_file=questions_file,
                answers_file=answers_file
            )
            
            # 读取生成的答案统计
            with open(answers_file, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
                self.stats["answers_generated"] = len(answers_data)
                
                # 统计校验结果
                for answer in answers_data:
                    validation_report = answer.get("validation_report", {})
                    if validation_report.get("validation_passed", False):
                        self.stats["validation_passed"] += 1
                    else:
                        self.stats["validation_failed"] += 1
            
            print(f"✓ 答案生成完成: {len(answers_data)} 个答案")
            print(f"  - 校验通过: {self.stats['validation_passed']}")
            print(f"  - 校验未通过: {self.stats['validation_failed']}")
            
        except Exception as e:
            print(f"✗ 答案生成失败: {e}")
            self.stats["errors"].append(f"答案生成失败: {str(e)}")
            raise
        
        # Step 3: 生成最终数据集和统计信息
        print("\n" + "=" * 80)
        print("Step 3: 生成最终数据集")
        print("=" * 80)
        
        final_dataset_file = output_dir / f"vqa_dataset_{timestamp}.json"
        statistics_file = output_dir / f"statistics_{timestamp}.json"
        
        try:
            # 读取答案数据
            with open(answers_file, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            
            # 生成最终数据集（可以在这里进行额外的处理和过滤）
            final_dataset = self._prepare_final_dataset(answers_data)
            
            # 保存最终数据集
            with open(final_dataset_file, 'w', encoding='utf-8') as f:
                json.dump(final_dataset, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 最终数据集已保存: {final_dataset_file}")
            print(f"  - 总样本数: {len(final_dataset)}")
            
            # 生成统计信息
            statistics = self._generate_statistics(final_dataset, answers_data)
            statistics["pipeline_info"] = {
                "input_file": str(input_file),
                "timestamp": timestamp,
                "pipeline_names": pipeline_names,
                "max_samples": max_samples
            }
            statistics["stats"] = self.stats
            
            # 保存统计信息
            with open(statistics_file, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, ensure_ascii=False, indent=2)
            
            print(f"✓ 统计信息已保存: {statistics_file}")
            
        except Exception as e:
            print(f"✗ 生成最终数据集失败: {e}")
            self.stats["errors"].append(f"生成最终数据集失败: {str(e)}")
            raise
        
        # 生成摘要报告
        print("\n" + "=" * 80)
        print("流程完成摘要")
        print("=" * 80)
        self._print_summary(final_dataset_file, statistics_file, questions_file, answers_file)
        
        return {
            "final_dataset": final_dataset_file,
            "statistics": statistics_file,
            "questions": questions_file,
            "answers": answers_file,
            "stats": self.stats
        }
    
    def _prepare_final_dataset(self, answers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        准备最终数据集
        
        可以在这里进行额外的处理和过滤
        """
        final_dataset = []
        
        for answer in answers_data:
            # 构建最终数据项
            item = {
                # 基本信息
                "id": answer.get("id"),
                "sample_index": answer.get("sample_index"),
                "source_a_id": answer.get("source_a_id"),
                
                # 问题和答案
                "question": answer.get("question"),
                "full_question": answer.get("full_question"),
                "answer": answer.get("answer"),
                "question_type": answer.get("question_type"),
                
                # 图片
                "image_base64": answer.get("image_base64"),
                
                # 选择题特有字段
                "options": answer.get("options"),
                "correct_option": answer.get("correct_option"),
                
                # 解释
                "explanation": answer.get("explanation", ""),
                
                # Pipeline信息
                "pipeline_name": answer.get("pipeline_name"),
                "pipeline_intent": answer.get("pipeline_intent"),
                "answer_type": answer.get("answer_type"),
                
                # 校验信息（简化版）
                "validation_passed": answer.get("validation_report", {}).get("validation_passed", False),
                "validation_score": self._calculate_validation_score(answer.get("validation_report", {})),
                
                # 时间戳
                "generated_at": answer.get("generated_at", "")
            }
            
            final_dataset.append(item)
        
        return final_dataset
    
    def _calculate_validation_score(self, validation_report: Dict[str, Any]) -> float:
        """
        计算校验评分
        
        基于格式检查和VQA验证结果计算综合评分
        """
        if not validation_report:
            return 0.0
        
        score = 0.0
        
        # 格式检查评分（40%）
        format_check = validation_report.get("format_check", {})
        if format_check.get("passed", False):
            score += 0.4
        
        # VQA验证评分（60%）
        vqa_validation = validation_report.get("vqa_validation", {})
        
        # 困惑度分析（20%）
        perplexity = vqa_validation.get("perplexity_analysis", {})
        if perplexity.get("passed", False):
            clarity_score = perplexity.get("clarity_score", 0.5)
            score += 0.2 * clarity_score
        
        # 置信度评估（20%）
        confidence = vqa_validation.get("confidence_assessment", {})
        if confidence.get("passed", False):
            conf_score = confidence.get("confidence", 0.5)
            score += 0.2 * conf_score
        
        # 答案验证（20%）
        answer_validation = vqa_validation.get("answer_validation", {})
        if answer_validation.get("passed", False):
            score += 0.2
        
        return round(score, 3)
    
    def _generate_statistics(self, final_dataset: List[Dict[str, Any]], answers_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成统计信息
        """
        stats = {
            "total_samples": len(final_dataset),
            "by_question_type": {},
            "by_pipeline": {},
            "validation_summary": {
                "passed": 0,
                "failed": 0,
                "average_score": 0.0
            },
            "quality_metrics": {}
        }
        
        # 按题型统计
        for item in final_dataset:
            qtype = item.get("question_type", "unknown")
            stats["by_question_type"][qtype] = stats["by_question_type"].get(qtype, 0) + 1
        
        # 按pipeline统计
        for item in final_dataset:
            pipeline = item.get("pipeline_name", "unknown")
            stats["by_pipeline"][pipeline] = stats["by_pipeline"].get(pipeline, 0) + 1
        
        # 校验摘要
        validation_scores = []
        for item in final_dataset:
            if item.get("validation_passed", False):
                stats["validation_summary"]["passed"] += 1
            else:
                stats["validation_summary"]["failed"] += 1
            
            score = item.get("validation_score", 0.0)
            if score > 0:
                validation_scores.append(score)
        
        if validation_scores:
            stats["validation_summary"]["average_score"] = round(
                sum(validation_scores) / len(validation_scores), 3
            )
        
        # 质量指标
        stats["quality_metrics"] = {
            "has_explanation": sum(1 for item in final_dataset if item.get("explanation")),
            "has_image": sum(1 for item in final_dataset if item.get("image_base64")),
            "complete_options": sum(1 for item in final_dataset 
                                  if item.get("question_type") == "multiple_choice" 
                                  and item.get("options") and len(item.get("options", {})) >= 2)
        }
        
        return stats
    
    def _print_summary(
        self,
        final_dataset_file: Path,
        statistics_file: Path,
        questions_file: Path,
        answers_file: Path
    ):
        """打印摘要报告"""
        print(f"输入记录数: {self.stats['input_records']}")
        print(f"生成问题数: {self.stats['questions_generated']}")
        print(f"生成答案数: {self.stats['answers_generated']}")
        print(f"校验通过: {self.stats['validation_passed']}")
        print(f"校验未通过: {self.stats['validation_failed']}")
        print()
        print("输出文件:")
        print(f"  - 最终数据集: {final_dataset_file}")
        print(f"  - 统计信息: {statistics_file}")
        if questions_file:
            print(f"  - 问题文件: {questions_file}")
        if answers_file:
            print(f"  - 答案文件: {answers_file}")
        print()
        if self.stats["errors"]:
            print("错误信息:")
            for error in self.stats["errors"]:
                print(f"  - {error}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='VQA数据集生成完整流程 - 从batch_process.sh输出生成完整VQA数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本使用
  python generate_vqa/pipeline.py input.json output_dir/
  
  # 指定pipeline和样本数
  python generate_vqa/pipeline.py input.json output_dir/ \\
      --pipelines question object_counting \\
      -n 100
  
  # 不保存中间结果
  python generate_vqa/pipeline.py input.json output_dir/ --no-intermediate
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入JSON文件路径（batch_process.sh的输出）'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='输出目录路径'
    )
    parser.add_argument(
        '--question-config',
        type=str,
        default=None,
        help='问题生成配置文件路径（默认: generate_vqa/question_config.json）'
    )
    parser.add_argument(
        '--answer-config',
        type=str,
        default=None,
        help='答案生成配置文件路径（默认: generate_vqa/generate_answer/answer_config.json）'
    )
    parser.add_argument(
        '--pipelines',
        type=str,
        nargs='+',
        default=None,
        help='要使用的pipeline列表（默认: 使用所有pipeline）'
    )
    parser.add_argument(
        '-n', '--max-samples',
        type=int,
        default=None,
        help='最大处理样本数（默认: 全部）'
    )
    parser.add_argument(
        '--no-intermediate',
        action='store_true',
        help='不保存中间结果（问题和答案文件）'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    # 处理配置文件路径（支持相对路径和绝对路径）
    question_config_path = None
    if args.question_config:
        question_config_path = Path(args.question_config)
        if not question_config_path.is_absolute():
            # 如果是相对路径，尝试相对于当前工作目录和项目根目录
            cwd_path = Path.cwd() / question_config_path
            project_root = Path(__file__).parent.parent
            project_path = project_root / question_config_path
            if cwd_path.exists():
                question_config_path = cwd_path
            elif project_path.exists():
                question_config_path = project_path
            # 如果都不存在，保持原路径，让后续代码处理
    
    answer_config_path = None
    if args.answer_config:
        answer_config_path = Path(args.answer_config)
        if not answer_config_path.is_absolute():
            # 如果是相对路径，尝试相对于当前工作目录和项目根目录
            cwd_path = Path.cwd() / answer_config_path
            project_root = Path(__file__).parent.parent
            project_path = project_root / answer_config_path
            if cwd_path.exists():
                answer_config_path = cwd_path
            elif project_path.exists():
                answer_config_path = project_path
            # 如果都不存在，保持原路径，让后续代码处理
    
    if not input_file.exists():
        print(f"[ERROR] 输入文件不存在: {input_file}")
        print(f"  当前工作目录: {Path.cwd()}")
        return 1
    
    # 检查配置文件是否存在（如果指定了）
    if question_config_path and not question_config_path.exists():
        print(f"[WARNING] 问题配置文件不存在: {question_config_path}")
        print(f"  将使用默认配置文件")
        question_config_path = None
    
    if answer_config_path and not answer_config_path.exists():
        print(f"[WARNING] 答案配置文件不存在: {answer_config_path}")
        print(f"  将使用默认配置文件")
        answer_config_path = None
    
    try:
        # 初始化流程
        pipeline = VQAPipeline(
            question_config_path=question_config_path,
            answer_config_path=answer_config_path
        )
        
        # 运行流程
        result = pipeline.run(
            input_file=input_file,
            output_dir=output_dir,
            pipeline_names=args.pipelines,
            max_samples=args.max_samples,
            save_intermediate=not args.no_intermediate
        )
        
        print("\n✓ 流程执行成功！")
        
    except Exception as e:
        print(f"\n✗ 流程执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

