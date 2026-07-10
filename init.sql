-- ==========================================
-- 客户分析与智能查询系统 — 数据库初始化
-- 架构: MCP 三表分层 (Base / Summary / Analytics)
-- 兼容: MySQL 8.0+ / SQLite 3.x
-- ==========================================

CREATE DATABASE IF NOT EXISTS `airline_analytics`
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE `airline_analytics`;

-- ==========================================
-- 1. 客户基础信息表 (Dimension)
-- ==========================================
CREATE TABLE IF NOT EXISTS `customer_base` (
    `member_no`      VARCHAR(20)   NOT NULL COMMENT '会员编号(业务主键)',
    `age`            TINYINT       DEFAULT NULL COMMENT '年龄',
    `gender`         VARCHAR(10)   DEFAULT NULL COMMENT '性别(男/女/未知)',
    `ffp_tier`       TINYINT       DEFAULT NULL COMMENT '会员等级',
    `avg_discount`   DECIMAL(5,2)  DEFAULT NULL COMMENT '平均折扣率',
    `ffp_date`       DATE          DEFAULT NULL COMMENT '入会日期',
    `first_flight`   DATE          DEFAULT NULL COMMENT '首次飞行日期',
    `create_time`    DATETIME      DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`    DATETIME      DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`member_no`),
    INDEX `idx_gender_tier` (`gender`, `ffp_tier`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户基础信息维度表';

-- ==========================================
-- 2. 客户飞行事实汇总表 (DWS)
-- ==========================================
CREATE TABLE IF NOT EXISTS `customer_flight_summary` (
    `member_no`      VARCHAR(20)   NOT NULL COMMENT '会员编号(逻辑外键)',
    `flight_count`   INT           DEFAULT 0 COMMENT '总飞行次数(F特征)',
    `seg_km_sum`     INT           DEFAULT 0 COMMENT '总飞行里程(M特征)',
    `bp_sum`         INT           DEFAULT 0 COMMENT '总基本积分',
    `last_flight`    DATE          DEFAULT NULL COMMENT '最后飞行日期',
    `recency`        INT           DEFAULT NULL COMMENT '距观察期末天数(R特征)',
    `create_time`    DATETIME      DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    `update_time`    DATETIME      DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (`member_no`),
    INDEX `idx_recency` (`recency`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户飞行行为事实汇总表';

-- ==========================================
-- 3. 客户 AI 分析结果表 (ADS)
-- ==========================================
CREATE TABLE IF NOT EXISTS `customer_analytics` (
    `member_no`      VARCHAR(20)   NOT NULL COMMENT '会员编号(逻辑外键)',
    `r_score`        TINYINT        DEFAULT NULL COMMENT '最近消费评分(1-5)',
    `f_score`        TINYINT        DEFAULT NULL COMMENT '消费频率评分(1-5)',
    `m_score`        TINYINT        DEFAULT NULL COMMENT '消费金额评分(1-5)',
    `rfm_total`      TINYINT        DEFAULT NULL COMMENT 'RFM总分(3-15)',
    `cluster`        TINYINT        DEFAULT NULL COMMENT '聚类簇(0-3)',
    `value_label`    VARCHAR(20)    DEFAULT NULL COMMENT '客户价值标签',
    `update_time`    DATETIME       DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '算法计算更新时间',
    PRIMARY KEY (`member_no`),
    INDEX `idx_cluster` (`cluster`),
    INDEX `idx_rfm_label` (`rfm_total`, `value_label`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客户RFM与聚类智能分析结果表';
