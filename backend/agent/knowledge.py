"""
Cơ sở kiến thức tóm tắt cho 14 nhãn NIH + Normal/Other.
Dùng làm tool `lookup_pathology_info`. Không thay thế UpToDate / sách giáo khoa.
"""
from __future__ import annotations

PATHOLOGY_INFO: dict[str, dict] = {
    "Normal": {
        "description": "Không có dấu hiệu bất thường rõ ràng trên X-quang.",
        "common_findings": ["Phổi sáng đều", "Tim trong giới hạn bình thường"],
        "next_steps": ["Khám định kỳ", "Theo dõi triệu chứng nếu có"],
    },
    "Other": {
        "description": "Bất thường chưa phân loại cụ thể trong nhóm chính.",
        "common_findings": ["Bóng mờ không đặc hiệu"],
        "next_steps": ["Hội chẩn bác sĩ X-quang", "Có thể cần CT scan"],
    },
    "Atelectasis": {
        "description": "Xẹp một phần hoặc toàn bộ thùy phổi.",
        "common_findings": ["Vùng mờ", "Trung thất kéo lệch", "Cơ hoành nâng"],
        "next_steps": ["Xác định nguyên nhân tắc nghẽn", "CT ngực nếu nghi ngờ u"],
    },
    "Cardiomegaly": {
        "description": "Bóng tim to (chỉ số tim/lồng ngực > 0.5 trên phim PA).",
        "common_findings": ["Cardiothoracic ratio > 50%", "Bóng tim mở rộng hai bờ"],
        "next_steps": ["Siêu âm tim", "ECG", "Đánh giá suy tim"],
    },
    "Effusion": {
        "description": "Tràn dịch màng phổi.",
        "common_findings": ["Mờ góc sườn hoành", "Đường cong Damoiseau"],
        "next_steps": ["Siêu âm màng phổi", "Chọc dò xét nghiệm dịch nếu chỉ định"],
    },
    "Infiltration": {
        "description": "Thâm nhiễm phổi không đặc hiệu (viêm, xuất huyết, u…).",
        "common_findings": ["Bóng mờ lan toả", "Phế quản đồ khí"],
        "next_steps": ["Xét nghiệm máu", "CT scan nếu lâm sàng nặng"],
    },
    "Mass": {
        "description": "Khối ≥ 3cm trong phổi — nghi ngờ u nguyên phát hoặc di căn.",
        "common_findings": ["Bóng tròn/đa cung", "Bờ tia gai", "Có thể vôi hoá"],
        "next_steps": ["CT ngực có cản quang", "Hội chẩn ung bướu", "PET-CT nếu nghi di căn"],
    },
    "Nodule": {
        "description": "Nốt phổi đơn độc < 3cm.",
        "common_findings": ["Bóng tròn nhỏ", "Bờ rõ hoặc mờ"],
        "next_steps": ["CT theo dõi (Fleischner guidelines)", "Sinh thiết nếu nguy cơ cao"],
    },
    "Pneumonia": {
        "description": "Viêm phổi (vi khuẩn, virus, nấm).",
        "common_findings": ["Đông đặc thuỳ", "Phế quản đồ khí", "Có thể tràn dịch kèm"],
        "next_steps": ["Cấy đờm/máu", "Kháng sinh theo phác đồ", "Theo dõi sau 2 tuần"],
    },
    "Pneumothorax": {
        "description": "Tràn khí màng phổi.",
        "common_findings": ["Đường viền lá tạng phổi", "Mất vân phổi vùng ngoại vi"],
        "next_steps": ["Đánh giá kích thước", "Dẫn lưu màng phổi nếu cần", "Theo dõi sát"],
    },
    "Consolidation": {
        "description": "Đông đặc nhu mô phổi (phế nang chứa dịch/mủ/máu).",
        "common_findings": ["Bóng mờ đồng nhất", "Phế quản đồ khí"],
        "next_steps": ["Phân biệt viêm phổi vs xuất huyết phế nang", "Cấy đờm"],
    },
    "Edema": {
        "description": "Phù phổi — thường do tim trái suy.",
        "common_findings": ["Bóng mờ cánh bướm", "Đường Kerley B", "Bóng tim to"],
        "next_steps": ["Siêu âm tim", "Pro-BNP", "Lợi tiểu nếu suy tim"],
    },
    "Emphysema": {
        "description": "Khí phế thũng — phá huỷ vách phế nang.",
        "common_findings": ["Phổi tăng sáng", "Cơ hoành phẳng", "Khoang liên sườn rộng"],
        "next_steps": ["Hô hấp ký", "Bỏ thuốc lá", "CT ngực nếu cần phân loại"],
    },
    "Fibrosis": {
        "description": "Xơ hoá phổi — tổn thương kẽ tiến triển.",
        "common_findings": ["Bóng lưới", "Tổ ong", "Co rút thể tích phổi"],
        "next_steps": ["HRCT", "Thăm dò chức năng phổi", "Hội chẩn hô hấp"],
    },
    "Pleural_Thickening": {
        "description": "Dày màng phổi — sẹo cũ, lao, bệnh nghề nghiệp.",
        "common_findings": ["Đường viền màng phổi đậm", "Vôi hoá màng phổi"],
        "next_steps": ["Khai thác phơi nhiễm asbestos/lao cũ"],
    },
    "Hernia": {
        "description": "Thoát vị hoành (gồm hiatal hernia).",
        "common_findings": ["Bóng khí sau bóng tim", "Mức nước-khí trên cơ hoành"],
        "next_steps": ["Chụp tương phản tiêu hoá trên", "Hội chẩn ngoại khoa nếu lớn"],
    },
}


def lookup_pathology_info(label: str) -> dict:
    info = PATHOLOGY_INFO.get(label)
    if info is None:
        return {
            "description": f"Chưa có dữ liệu chi tiết cho '{label}'.",
            "common_findings": [],
            "next_steps": ["Hội chẩn bác sĩ chuyên khoa"],
        }
    return {**info, "label": label}
